import streamlit as st
import torch
from transformers import AutoTokenizer, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence
from Bio.Seq import Seq
from Bio.Data import CodonTable
import RNA
import matplotlib.pyplot as plt
import forgi
import forgi.visual.mplotlib as fvm
import io

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer").to(DEVICE)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Human codon table and synonymous codons dictionary
human_table = CodonTable.unambiguous_dna_by_name["Standard"]
synonymous_codons = {aa: [codon for codon, a in human_table.forward_table.items() if a == aa] for aa in human_table.forward_table.values()}

def translate_mrna_to_protein(mrna_sequence):
    coding_dna = Seq(mrna_sequence)
    protein = coding_dna.translate()
    return str(protein)

def humanize_sequence(input_sequence, sequence_type):
    if sequence_type == "mRNA":
        protein_sequence = translate_mrna_to_protein(input_sequence)
    else:
        protein_sequence = input_sequence
    
    output = predict_dna_sequence(
        protein=protein_sequence,
        organism="Homo sapiens",
        device=DEVICE,
        tokenizer_object=tokenizer,
        model_object=model,
        attention_type="original_full",
    )
    
    return output.predicted_dna

def get_wobble_codon(codon):
    aa = human_table.forward_table.get(codon, 'X')
    a_ending_codons = [c for c in synonymous_codons.get(aa, []) if c[-1] == 'A']
    t_ending_codons = [c for c in synonymous_codons.get(aa, []) if c[-1] == 'T']

    if codon[-1] == 'A':
        return codon
    elif a_ending_codons:
        return a_ending_codons[0]
    elif codon[-1] == 'T':
        return codon
    elif t_ending_codons:
        return t_ending_codons[0]
    else:
        return codon

def apply_wobble_substitution(dna_sequence):
    return ''.join(get_wobble_codon(dna_sequence[i:i+3]) for i in range(0, len(dna_sequence), 3))

def optimize_sequence(input_sequence, sequence_type, humanize, wobble):
    if humanize:
        optimized_dna = humanize_sequence(input_sequence, sequence_type)
    else:
        optimized_dna = input_sequence if sequence_type == "mRNA" else Seq(input_sequence).transcribe().back_transcribe()
    
    if wobble:
        optimized_dna = apply_wobble_substitution(optimized_dna)
    
    return optimized_dna

def dna_to_rna(dna_sequence):
    return dna_sequence.replace('T', 'U')

def predict_rna_structure(rna_sequence):
    fc = RNA.fold_compound(rna_sequence)
    (ss, mfe) = fc.mfe()
    return ss, mfe

def create_fx_file(rna_sequence, structure, filename="temp.fx"):
    with open(filename, "w") as f:
        f.write(f">temp_rna\n{rna_sequence}\n{structure}\n")
    return filename

def visualize_rna_structure(rna_sequence, structure):
    # Create fx file
    fx_file = create_fx_file(rna_sequence, structure)
    
    # Load RNA structure
    cg = forgi.load_rna(fx_file, allow_many=False)
    
    # Create figure and plot RNA
    fig, ax = plt.subplots(figsize=(70, 70))
    fvm.plot_rna(cg, ax=ax, text_kwargs={"fontweight":"black"}, lighten=0.2,
                 backbone_kwargs={"linewidth":1})
    
    #ax.set_title("RNA Secondary Structure")
    ax.axis('off')
    
    # Remove the temporary fx file
    import os
    os.remove(fx_file)
    
    return fig

st.title("CGInvites mRNA Builder")

sequence_type = st.selectbox("Select input sequence type:", ["Protein", "mRNA"])
input_sequence = st.text_area(f"Enter your {sequence_type} sequence:")
optimization_methods = st.multiselect(
    "Select optimization method(s):",
    ["Humanization", "Wobble substitution"],
    default=["Humanization", "Wobble substitution"]  # Changed this line to select all options by default
)

if st.button("Optimize Sequence"):
    if input_sequence:
        humanize = "Humanization" in optimization_methods
        wobble = "Wobble substitution" in optimization_methods
        
        optimized_dna = optimize_sequence(input_sequence, sequence_type, humanize, wobble)
        optimized_rna = dna_to_rna(optimized_dna)
        
        st.success("Optimization complete!")
        
        st.subheader("Optimized DNA Sequence:")
        st.code(optimized_dna)
        
        st.subheader("Optimized RNA Sequence:")
        st.code(optimized_rna)
        
        # RNA secondary structure prediction
        ss, mfe = predict_rna_structure(optimized_rna)
        st.subheader("Predicted RNA Secondary Structure:")
        st.code(ss)
        st.write(f"Minimum Free Energy: {mfe:.2f} kcal/mol")
        
        ## RNA secondary structure visualization
        #fig = visualize_rna_structure(optimized_rna, ss)
        #st.pyplot(fig)
    else:
        st.warning("Please enter a sequence.")
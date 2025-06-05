import numpy as np
import pandas as pd
from Bio import SeqIO


def cgr(sequence, img_size=32):
    """
    Generate a CGR image for a DNA sequence.
    Args:
        sequence (str): DNA sequence (A, C, G, T).
        img_size (int): Output image size (img_size x img_size).
    Returns:
        np.ndarray: CGR grayscale image.
    """
    # Map nucleotides to corners of unit square
    corners = {'A': (0, 0), 'C': (0, 1), 'G': (1, 1), 'T': (1, 0)}
    
    # Start at center
    x, y = 0.5, 0.5
    
    img = np.zeros((img_size, img_size)) # type: ignore
    for base in sequence.upper():
        if base not in corners:
            # Skip kmer containing N
            continue 
        cx, cy = corners[base]
        # Midpoint to corner
        x = (x + cx) / 2
        y = (y + cy) / 2
        # Map point to pixel coordinates
        px = int(x * (img_size - 1))
        py = int(y * (img_size - 1))
        img[py, px] += 1

    # Normalize image
    img = img / img.max() if img.max() > 0 else img
    return img

def kmer_to_index(kmer):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    index = 0
    for i, nucleotide in enumerate(kmer):
        index = index * 4 + mapping[nucleotide]
    return index

def fcgr(sequence, k):
    dim = 2 ** k
    matrix = np.zeros((dim, dim), dtype=np.float32)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if not all(n in 'ACGT' for n in kmer):
            # Skip kmer containing N
            continue 
        index = kmer_to_index(kmer)
        x = index // dim
        y = index % dim
        matrix[x, y] += 1

    matrix /= matrix.max()
    return matrix

def split_sequence_to_kmers(sequence, k_mer):
    return [sequence[i:i+k_mer] for i in range(len(sequence) - k_mer + 1)]


def load_genome_data(filepath="../dataset/16S_AMP.fasta",  label_path ="../dataset/taxonomy.csv",classify_at_level = "GENUS"):
    
    label_dict={}
    label_count = 0
    taxonomy_df = pd.read_csv(label_path, index_col=0)

    df = pd.DataFrame()
    with open(filepath) as handle:
        seqIt =  SeqIO.parse(handle, "fasta")
        while (sequence := next(seqIt, None)) is not None:
            genome_id = str(sequence.id).split("_")[1]
            key = str(taxonomy_df.loc[str(genome_id), classify_at_level]).strip()
            if (value := label_dict.get(key)) is None:
                value = label_count
                label_dict.update({key: value})
                label_count+=1
            entry = {
                    "sequence" : [str(sequence.seq)],
                    "label" : value,
                    "genome_id" : genome_id
                    #  "label" : [int(str(taxonomy_df.loc[str(sequence.id), classify_at_level]).strip() == target)]
                    }
            row = pd.DataFrame.from_dict(entry)
            df = pd.concat([df, row], axis=0)
    return df
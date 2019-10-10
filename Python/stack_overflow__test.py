from Bio import SeqIO
arch = 'DENV1.gb'
record = SeqIO.parse(arch, 'genbank')
rec = next(record)

print("test")

for f in rec.features:
    if f.type == 'gene':
        print(f.qualifiers['gene'], f.location)

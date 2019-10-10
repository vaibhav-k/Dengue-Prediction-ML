from Bio import SeqIO

for record in SeqIO.parse("../../Data/Metadata/DENV2/DENV2.gb", "genbank"):
	print(record.id)

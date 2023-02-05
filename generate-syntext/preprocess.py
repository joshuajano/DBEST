orig_path = '/media/verihubs/verihubs-2TB/datasets/text-scene/SRNet-nw-dataset/SRNet-nw/SRNet-Datagen/Synthtext/data/texts.txt'

file = open(orig_path,"r")
all_texts = file.readlines()
file.close()
texts_with_4_chars = []
for text in all_texts:
    cln_text = text.replace("\n", "")
    if len(cln_text)==4 or len(cln_text)==5 or len(cln_text)==3 :
        texts_with_4_chars.append(cln_text)

save_path = '345_char_texts.txt'
with open(save_path, 'w') as f:
    for line in texts_with_4_chars:
        f.write(line)
        f.write('\n')
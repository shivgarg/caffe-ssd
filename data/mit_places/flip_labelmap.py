
f=open('labelmap_oi.prototxt')
g=open('labelmap_oi_flip.prototxt','w')

string = ['', '', '', '', '']
cnt = 0
for lines in f:
	if 'item {' in lines:
		string[1],string[3] = string[3], string[1]
		string[2] = string[2].replace('id','label')
#		string[1] = string[1].replace(' ','')
		g.write(''.join(string))
		cnt = 1
		string[0] = lines
	else:
		string[cnt] = lines
		cnt +=1

string[2] = string[2].replace('id','label')
string[1],string[3] = string[3], string[1]
g.write(''.join(string))

		

import matplotlib.pyplot as plt
import csv

output_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/code/extra"
x = []
y = []
z = []
fig, ax = plt.subplots()

# with open('/extra/cca_ov.csv','r') as csvfile:
with open(f'{output_path}/cca+smig.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	for row in lines:
		x.append(row[0])
		y.append(float(row[1]))
		z.append(float(row[2]))

ax.plot(x, y, color = 'b', linestyle = 'dashed',
		marker = 'o',label = "CCA(x_d,x_f-ov)")

ax.plot(x, z, color = 'r', linestyle = 'dashed',
		marker = 'o',label = "SMIG(x_d,x_e,x_f-ov)")

plt.xlabel('Components/ Projections')
plt.ylabel('AUC')
plt.grid()
plt.legend()
plt.tight_layout()     
plt.savefig(output_path+'/cca+smig_ov.png')  

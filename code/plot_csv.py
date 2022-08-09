import matplotlib.pyplot as plt
import csv
import argparse
from sklearn.metrics import auc

output_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/code/extra/"
# x,y,z = [],[],[]
x1,x2,x3,x4,y1,y2,y3,y4 = [],[],[],[],[],[],[],[]
fig,ax = plt.subplots()

# Parse model, imputer, neighbors(KNN) values from user
parser = argparse.ArgumentParser()
parser.add_argument('--name1', type=str)
parser.add_argument('--name2', type=str)
parser.add_argument('--name3', type=str)
parser.add_argument('--name4', type=str)
parser.add_argument('--filename', type=str)
args = parser.parse_args()

name1,name2,name3,name4,filename = args.name1,args.name2,args.name3,args.name4,args.filename

# # with open('/extra/cca_ov.csv','r') as csvfile:
# with open(f'{output_path}/cca+smig.csv','r') as csvfile:
# 	lines = csv.reader(csvfile, delimiter=',')
# 	for row in lines:
# 		x.append(row[0])
# 		y.append(float(row[1]))
# 		z.append(float(row[2]))

with open(f'{output_path}'+name1+'.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
    # next(lines)
	for row in lines:
		x1.append(float(row[1]))
		y1.append(float(row[2]))

with open(f'{output_path}'+name2+'.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
    # next(lines)
	for row in lines:
		x2.append(float(row[1]))
		y2.append(float(row[2]))

with open(f'{output_path}'+name3+'.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
    # next(lines)
	for row in lines:
		x3.append(float(row[1]))
		y3.append(float(row[2]))

with open(f'{output_path}'+name4+'.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
    # next(lines)
	for row in lines:
		x4.append(float(row[1]))
		y4.append(float(row[2]))

# ax.plot(x, y, color = 'b', linestyle = 'dashed',
# 		marker = 'o',label = "CCA(x_d,x_f-ov)")

# ax.plot(x, z, color = 'r', linestyle = 'dashed',
# 		marker = 'o',label = "SMIG(x_d,x_e,x_f-ov)")

# auc1,auc2,auc3 = auc(x1, y1),auc(x2, y2),auc(x3, y3)
auc1,auc2,auc3,auc4 = auc(x1, y1),auc(x2, y2),auc(x3, y3),auc(x4, y4)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="y", label="Chance", alpha=0.8)

ax.plot(x1, y1, color = 'r',lw=2, linestyle="--",
		label = r"CCA (AUC = %0.3f)" % round(auc1,3))

ax.plot(x2, y2, color = 'g',lw=2, linestyle="--",
		label = r"SFS (AUC = %0.3f)" % round(auc2,3))

ax.plot(x3, y3, color = 'b',lw=2, linestyle="--",
		label = r"SFS_thresh (AUC = %0.3f)" % round(auc3,3))

ax.plot(x4, y4, color = 'k',lw=2,
		label = r"CCA+SFS_thresh (AUC = %0.3f)" % round(auc4,3))

# plt.xlabel('Components/ Projections')
# plt.ylabel('AUC')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
# plt.grid()
plt.legend()
ax.legend(loc="lower right")
plt.tight_layout()     
plt.savefig(output_path+filename+'.png')  
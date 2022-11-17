from fileinput import filename
import matplotlib.pyplot as plt
import csv
import argparse
from sklearn.metrics import auc,roc_auc_score
from astropy.stats import binom_conf_interval
import numpy as np
from metrics import stat_util
import pickle

output_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/code/extra/"
x1,x2,x3,x4,y1,y2,y3,y4 = [],[],[],[],[],[],[],[]
fold=True
# x,y,z = [],[],[]

# Parse model, imputer, neighbors(KNN) values from user
if fold:
	parser = argparse.ArgumentParser()
	parser.add_argument('--options', type=str)
	parser.add_argument('--name1', type=str)
	parser.add_argument('--name2', type=str)
	parser.add_argument('--name3', type=str, default='None')
	parser.add_argument('--name4', type=str, default='None')
	parser.add_argument('--filename', type=str)
	args = parser.parse_args()
	filename=args.filename

# # with open('/extra/cca_ov.csv','r') as csvfile:
# with open(f'{output_path}/cca+smig.csv','r') as csvfile:
# 	lines = csv.reader(csvfile, delimiter=',')
# 	for row in lines:
# 		x.append(row[0])
# 		y.append(float(row[1]))
# 		z.append(float(row[2]))
if args.options=='save':
	name1,name2,name3,name4 = args.name1,args.name2,args.name3,args.name4

	fig,ax = plt.subplots()
	with open(f'{output_path}'+name1+'.csv','r') as csvfile:
		lines = csv.reader(csvfile, delimiter=',')
		# next(lines)
		for row in lines:
			x1.append(float(row[0]))
			y1.append(float(row[1]))

	with open(f'{output_path}'+name2+'.csv','r') as csvfile:
		lines = csv.reader(csvfile, delimiter=',')
		# next(lines)
		for row in lines:
			x2.append(float(row[0]))
			y2.append(float(row[1]))

	# with open(f'{output_path}'+name3+'.csv','r') as csvfile:
	# 	lines = csv.reader(csvfile, delimiter=',')
	# 	# next(lines)
	# 	for row in lines:
	# 		x3.append(float(row[1]))
	# 		y3.append(float(row[2]))

	# with open(f'{output_path}'+name4+'.csv','r') as csvfile:
	# 	lines = csv.reader(csvfile, delimiter=',')
	# 	# next(lines)
	# 	for row in lines:
	# 		x4.append(float(row[1]))
	# 		y4.append(float(row[2]))

	# ax.plot(x, y, color = 'b', linestyle = 'dashed',
	# 		marker = 'o',label = "CCA(x_d,x_f-ov)")

	# ax.plot(x, z, color = 'r', linestyle = 'dashed',
	# 		marker = 'o',label = "SMIG(x_d,x_e,x_f-ov)")

	# auc1,auc2,auc3,auc4  = auc(x1, y1),auc(x2, y2),auc(x3, y3),auc(x4, y4)

	# ax.plot([0, 1], [0, 1], linestyle="-.", lw=1, color="r", label="Chance", alpha=0.8)

	ax.plot(x1, y1, color = 'r',lw=2, linestyle="--",
			marker = 'o', markersize=9,label = "RECC-SFS")
			# label = r"SFS (AUC=%0.2f)" % round(auc1,2))

	ax.plot(x2, y2, color = 'b',lw=2, linestyle="--",
			marker = 'o', markersize=9,label = "RECC-SMIG")
			# label = r"RECC-SFS (AUC=%0.2f)" % round(auc2,2))

	# ax.plot(x3, y3, color = 'm',lw=2, linestyle="--",
			# label = r"CCA (AUC=%0.2f)" % round(auc3,2))

	# ax.plot(x4, y4, color = 'k',lw=2,
			# label = r"IDSF-CCA-RECC-SFS (AUC=%0.2f)" % round(auc4,2))

	# # Using Binomial conf intervals, as laid out in Sourati 2015
	# x4,y4=np.array(x4),np.array(y4)
	# [tprs_upper, tprs_lower] = binom_conf_interval(y4*48, 48, confidence_level=0.95, interval='wilson')  
	# tp_low,tp_up=np.array(tprs_lower),np.array(tprs_upper)

	# ax.fill_between(
	# 	x4,
	# 	tp_low,
	# 	tp_up,
	# 	color="grey",
	# 	alpha=0.2,
	# 	# label=r"$\pm$ 1 std. dev.",            
	# 	label=r'95% level of confidence',
	# )

	plt.xlabel(r'$\rho$')
	plt.ylabel('AUC')
	# plt.xlabel('1-Specificity')
	# plt.ylabel('Sensitivity')
	# plt.gca().set_aspect("equal")
	# plt.grid()
	plt.legend()
	ax.legend(loc="upper right")

	# plt.xlabel('Components/ Projections')
	# plt.ylabel('AUC')
	SMALL_SIZE = 8
	MEDIUM_SIZE = 10
	BIGGER_SIZE = 20

	plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	plt.tight_layout()     

	with open(f'{output_path}{filename}.pkl','wb') as fid:
		pickle.dump(ax, fid)
	# plt.show()
	plt.savefig(f'{output_path}{filename}.png')  

elif args.options=='stat':
	count=0
	with open(f'{output_path}'+name1+'.csv','r') as csvfile:
		lines = csv.reader(csvfile, delimiter=',')
		for row in lines:
			if count==0:
				continue
			x1.append(float(row[0]))
			y1.append(float(row[2]))
			count+=1

	count=0
	with open(f'{output_path}'+name2+'.csv','r') as csvfile:
		lines = csv.reader(csvfile, delimiter=',')
		for row in lines:
			if count==0:
				continue
			x2.append(float(row[0]))
			y2.append(float(row[2]))
			count+=1

	p, z = stat_util.pvalue(x2, y1, y2, score_fun=roc_auc_score)
	print('p:',p)
	# print('z:',z)

elif args.options=='load':
	with open(f'{output_path}{filename}','rb') as fid:
		ax = pickle.load(fid)
	plt.show()

# python plot_csv.py --name1 SFS_Ov_KNN_1_fixed_5_svm_rho --name2 SMIG_Ov_KNN_1_fixed_3_svm_rho --name3 CCA_Pos_KNN_1_fixed_7_roc_roc_pts --name4 CCA+SFS_Ov_KNN_1_fixed_7_best_cca+sfs_roc_roc_pts --filename recc-sfs-smig-rho

# python plot_csv.py --options load --filename recc-sfs-smig-rho.pkl

# python plot_csv.py --options save --name1 SFS_Ov_KNN_1_fixed_5_svm_rho --name2 SMIG_Ov_KNN_1_fixed_3_svm_rho --filename recc-sfs-smig-rho
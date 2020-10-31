# EpigeneticAgeML

1. data.csv

This file contains the information of 326 samples. All the sample IDs have been masked for de-identification.

The columns of the data file contain the following information:
SampleID	=	Pseudo sample ID
DNAmAge.10	=	DNA methylation age at 10 years
AgeAccelerationDiff.10	=	DNAmAge.10 - Age.10
AgeAccelerationResidual.10	=	Age acceleration residual at 10 years
Age.10	=	Age at 10 years
Gender.10	=	Sex at 10 years
AAHOAdjCellCounts.10	=	Intrinsic Epigenetic Age Acceleration (IEAA) at 10 years
HEIGHTCM_10	=	Height at 10 years (in cm)
BMI_10	=	BMI at 10 years  (Kg/m2)
WEIGHTKG_10	=	Weight at 10 years (in Kg)
EVERHADASTHMA_10	=	Ever had asthma before 10 years
ECZEMA_10	=	Ever had eczema before 10 years
HAYFEVER_10	=	Ever had heyfever before 10 years
FEV1_10	=	Force Expiratory Volume at 10 years
FVC_10	=	Force Vital Capacity at 10 years
FEV1BYFVC_10	=	FEV1/FVC at 10 years
DNAmAge.18	=	DNA methylation age at 18 years
AgeAccelerationDiff.18	=	DNAmAge.18 - Age.18
AgeAccelerationResidual.18	=	Age acceleration residual at 18 years
Age.18	=	Age at 18 years
Gender.18	=	Sex at 18 years
AAHOAdjCellCounts.18	=	Intrinsic Epigenetic Age Acceleration (IEAA) at 18 years
HEIGHTCM_18	=	Height at 18 years (in cm)
BMI_18	=	BMI at 18 years  (Kg/m2)
WEIGHTKG_18	=	Weight at 18 years (in Kg)
EVERHADASTHMA_18	=	Ever had asthma before 18 years
ECZEMA_18	=	Ever had eczema before 18 years
HAYFEVER_18	=	Ever had heyfever before 18 years
FEV1_18	=	Force Expiratory Volume at 18 years
FVC_18	=	Force Vital Capacity at 18 years
FEV1ByFVC.18	=	FEV1/FVC at 18 years
DOYOUCURRENTLYSMOKE_18	=	Smoking at 18 years


2. code.py
This file contains python code that developed the machine learning models and generated all the results based on Force Expiratory Volume (FEV1). FEV1 can be changed to FVC to achieve results for FVC.

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
# import matplotlib.pyplot as plt
import pandas as pd


def save_vol(save_path, tensor_3d, affine):
	"""
		save_path: path to write the volume to
		tensor_3d: 3D volume which needs to be saved
		affine: image orientation, translation 
	"""
	directory = os.path.dirname(save_path)

	if not os.path.exists(directory):
		os.makedirs(directory)
	
	volume = nib.Nifti1Image(tensor_3d, affine)
	volume.set_data_dtype(np.float32) 
	nib.save(volume, save_path)


def load_vol(load_path):
	"""
		load_path: volume path to load
		return:
			volume: loaded 3D volume
			affine: affine data specific to the volume
	"""
	if not os.path.exists(load_path):
		raise ValueError("path doesn't exist")

	nib_vol = nib.load(load_path)
	vol_data = nib_vol.get_data()
	vol_affine = nib_vol.affine
	return vol_data, vol_affine


def create_label_volume(label):
	"""
		label: classification label for Niftynet classification
	""" 

	label_volume = np.array([label], dtype="int32")
	label_volume = label_volume.reshape(1, 1, 1)
	return label_volume
	

def standardize_volume(volume, mask=None):
	"""
		volume: volume which needs to be normalized
		mask: brain mask, only required if you prefer not to
			consider the effect of air in normalization
	"""
	if mask != None: volume = volume*mask

	mean = np.mean(volume[volume != 0])
	std = np.std(volume[volume != 0])
	
	return (volume - mean)/std


def normalize_volume(volume, mask=None, _type='MinMax'):
	"""
		volume: volume which needs to be normalized
		mask: brain mask, only required if you prefer not to 
			consider the effect of air in normalization
		_type: {'Max', 'MinMax', 'Sum'}
	"""
	if mask != None: volume = mask*volume
	
	min_vol = np.min(volume)
	max_vol = np.max(volume)
	sum_vol = np.sum(volume)

	if _type == 'MinMax':
		return (volume - min_vol) / (max_vol - min_vol)
	elif _type == 'Max':
		return volume/max_vol
	elif _type == 'Sum':
		return volume/sum_vol
	else:
		raise ValueError("Invalid _type, allowed values are: {}".format('Max, MinMax, Sum'))
		

def resize_sitk_3D(original_volume, outputSize=None, interpolator=sitk.sitkLinear):
	"""
    		Resample 3D images Image:
    		For Labels use nearest neighbour interpolation
    		For image can use any: 
    			sitkNearestNeighbor = 1,
    			sitkLinear = 2,
			sitkBSpline = 3,
    			sitkGaussian = 4,
    			sitkLabelGaussian = 5, 
	"""
	volume = sitk.GetImageFromArray(original_volume) 
	inputSize = volume.GetSize()
	inputSpacing = volume.GetSpacing()
	outputSpacing = [1.0, 1.0, 1.0]

	if outputSize:
		# based on provided information and aspect ratio of the 
		# original volume
		outputSpacing[0] = inputSpacing[0] * (inputSize[0] /outputSize[0]);
		outputSpacing[1] = inputSpacing[1] * (inputSize[1] / outputSize[1]);
		outputSpacing[2] = inputSpacing[2] * (inputSize[2] / outputSize[2]);
	else:
		# If No outputSize is specified then resample to 1mm spacing
		outputSize = [0.0, 0.0, 0.0]
		outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
		outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
		outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)

	resampler = sitk.ResampleImageFilter()
	resampler.SetSize(outputSize)
	resampler.SetOutputSpacing(outputSpacing)
	resampler.SetOutputOrigin(volume.GetOrigin())
	resampler.SetOutputDirection(volume.GetDirection())
	resampler.SetInterpolator(interpolator)
	resampler.SetDefaultPixelValue(0)
	volume = resampler.Execute(volume)
	resampled_volume = sitk.GetArrayFromImage(volume)
	return resampled_volume



def random_patch_extraction(volume_path,
				label,
				save_root,
                type_, 
				number_patches, 
				patch_size, 
				normalization = False, 
				standardization = False):
	"""
		volume_path: path to fetch volume [str]
		save_root: root directory to save volumes [str]
        type_: anatomical or funcitonal type
		number_pathces: number of pathces expected to save [int]
		patch_size: size of volume [tuple(int)]
		normalization: boolean for volume normalization
		standardization: boolean for volume standardization
	"""

	dataset = volume_path.split("/")[7] 
	subject = volume_path.split("/")[-1][:-11]

	volume, affine = load_vol(volume_path)
	max_x, max_y, max_z = volume.shape
	
	if standardization: volume = standardize_volume(volume)
	if normalization: volume = normalize_volume(volume)

	if number_patches > 1:
		for i in range(number_pathces):
			x, y, z = np.random.randint(max_x), np.random.randint(max_y), np.randon.randint(max_z)
			cropped_volume = volume[x : x + patch_size[0], y : y + patch_size[1], z : z + patch_size[2]]
			save_path = os.path.join(save_root, dataset + "_" + subject + "_" + str(i), type_, "T1w.nii.gz")
			save_vol(save_path, cropped_volume, affine)

	elif number_patches == 1:
		resized_volume = resize_sitk_3D(volume, patch_size)
		save_path = os.path.join(save_root, dataset + "_" + subject, type_, "T1w_.nii.gz")
		save_vol(save_path, resized_volume, affine)

	else:
		raise ValueError("Number of patches should be greater than 1, but given value: {}".format(number_patches))



def process_images(data_root, save_root, verbose = True):
	"""
		data_root: root directory for all patient volumes
		save_root: directory to save all processed volumes
	"""
	datasets = os.listdir(data_root)
	for dataset in tqdm(datasets):
		if dataset in ["derivatives"]: continue

		dataset_path = os.path.join(data_root, dataset)
		
		print ("[INFO]", dataset_path)
	
		# Since few datasets don't have participants.tsv		
		try:
			subjects_df = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\t')
			participant_ids, sex = subjects_df['participant_id'].values, subjects_df['sex'].values
			subjects = [sub for sub in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, sub))]
		except: 
			continue
		
		# since the folder naming scheme followed in some datasets are differen 
		# when compared with participants_id in participants.tsv  

		subject_ids = []
		for id_ in participant_ids:
			for ii, subject in enumerate(subjects):
				if subject.__contains__(id_): subject_ids.append(subject)

	
		for subject, _label in tqdm(zip(subject_ids, sex)):
			
			if isinstance(_label, str): 
				label = 0 if _label.__contains__('M') else 1
			else: 
				continue # not sure what corresponds to what

			subject_anat_path = os.path.join(dataset_path, subject, 'anat', subject + '_T1w.nii.gz')
            subject_func_path = os.path.join(dataset_path, subject, 'func', subject + '_Bold.nii.gz')

			if os.path.exists(subject_path): 
				if verbose:
					print("[INFO] dataset: {}, subject: {}, path: {}".format(dataset, subject, subject_path))
				
				# to prevent permission issues
				try:
					random_patch_extraction(subject_path, label, save_root,
                        type_ = 'anat', 
						number_patches=16, 
						patch_size = (128, 128, 128), 
						normalization = True,
						standardization = True)
                    
                    random_patch_extraction(subject_path, label, save_root,
                        type_ = 'func', 
						number_patches=16, 
						patch_size = (128, 128, 128), 
						normalization = True,
						standardization = True)
				except: 
					continue
			else:
				seq_types = os.listdir(os.path.join(dataset_path, subject))
				for seq_type in seq_types:
					temp_path = os.path.join(dataset_path, subject, seq_type)
					if os.path.isdir(temp_path) and (not temp_path.__contains__('anat')):
						subject_path = os.path.join(temp_path, 'anat', subject	+ '_' + seq_type + '_T1w.nii.gz')
						if verbose: 
							print ("[INFO] dataset: {}, subject: {}, path: {}".format(dataset, subject, subject_path))

						# to prevent permission issues
						try:
							random_patch_extraction(subject_path, label, save_root,
                                    type_='anat',
									number_patches=1,
									patch_size = (128, 128, 128),
									normalization = True,
									standardization = True) 
                            
                            random_patch_extraction(subject_path, label, save_root,
                                    type_='func',
									number_patches=1,
									patch_size = (128, 128, 128),
									normalization = True,
									standardization = True) 
						except: 
							continue


def generate_csv(save_root, csv_root):
	"""
		save_root: data root directory
		csv_root: directory to save csv files
	"""
	subjects = os.listdir(save_root)
	
	subjects = [sub for sub in subjects if os.path.isdir(os.path.join(save_root, sub))]
	T1 = [os.path.join(save_root, sub, "T1w_.nii.gz") for sub in subjects]
	Label = [os.path.join(save_root, sub, "Label.nii.gz") for sub in subjects]
	
	df = pd.DataFrame()
	df['subjects'] = subjects
	df['paths'] = T1
	df.to_csv(os.path.join(csv_root, "T1_data.csv"), index=False, header=False)

	df = pd.DataFrame()
	df['subjects'] = subjects
	df['labels'] = Label
	df.to_csv(os.path.join(csv_root, "Label_data.csv"), index=False, header=False)

	temp = np.concatenate([np.array(['training']*int(0.7*len(subjects))),
				np.array(['validation']*int(0.2*len(subjects))), 
				np.array(['inference']*(len(subjects) - (int(0.2*len(subjects)) + int(0.7*len(subjects)))))])
	np.random.shuffle(temp)
	df = pd.DataFrame()
	df['subjects'] = subjects
	df['type'] = temp
	df.to_csv(os.path.join(csv_root, "cross_validation_fold_01.txt"), header = False, index = False)



if __name__ == "__main__":
	data_root = "/oak/stanford/groups/russpold/data/openneuro.org"
	save_root = "/oak/stanford/groups/russpold/data/openneuro.org/derivatives/kavinashscoolnet-1.0.0/processed_data"
	csv_root = "/oak/stanford/groups/russpold/data/openneuro.org/derivatives/kavinashscoolnet-1.0.0/csv_data"

	if not os.path.exists(save_root):
		os.system("mkdir -p " + save_root)
	if not os.path.exists(csv_root):
		os.system("mkdir -p " + csv_root)
	
	# process_images(data_root, save_root)
	generate_csv(save_root, csv_root)

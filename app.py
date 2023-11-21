import cv2, PIL, glob, os   
import numpy as np 
import matplotlib.pyplot as plt 

from flask import Flask, render_template, request
from keras.models import load_model 
from tensorflow.keras.utils import normalize 

from diffusers import StableDiffusionInpaintPipeline
import torch
# import gc

# gc.collect() 

# torch.cuda.empty_cache()

app = Flask(__name__)

# Model loading 
loaded_model_pretrained = load_model('./models/unet_mask.h5') 

def mask_image_creation(img_path): 
	image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
	image_resize = cv2.resize(image, (256, 256)) 
	image_normalize = np.expand_dims(normalize(np.array(image_resize), axis=1), 2) 
	image_gray = image_normalize[:,:,0][:,:,None] 
	image_expand = np.expand_dims(image_gray, axis=0) 
	# prediction with the model 
	predicted_mask = (loaded_model_pretrained.predict(image_expand)[0,:,:,0] > 0.2).astype(np.uint8) 
	return predicted_mask


def preprocessed_image(url_path):
  img_prepro = PIL.Image.open(url_path).convert("RGB").resize((256, 256))
  return img_prepro

def wrinkle_remove_using_stable_diffusion_2_inpaint(img_url, mask_url, streng, num_inferen):
	
	img = preprocessed_image(img_url)
	mask = preprocessed_image(mask_url)
 

	pipe = StableDiffusionInpaintPipeline.from_pretrained(
		"stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
	)

	#device = "cuda" if torch.cuda.is_available() else "cpu"
	pipe = pipe.to("cuda")   

	prompt = ""
	inpaint = pipe(prompt=prompt, image=img, mask_image=mask, strength=streng, num_inference_steps=num_inferen).images[0]
	return inpaint


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return render_template('image.html') 
    #return "this is the string text"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		all_path = ["./static/original_img", "./static/mask_img", "./static/inpainted_img"]
		for fol in all_path:
			all_file = os.listdir(fol) 
			for file_ in all_file:
				os.remove(os.path.join(fol, file_))  


		img_path = "./static/original_img/original.jpg" #+ img.filename	

		img.save(img_path)

		#p = predict_label(img_path) 
		predicted_mask_img = mask_image_creation(img_path)
		# generating name of the mask image 
		mask_path = "./static/mask_img/mask.jpg"  #+ img.filename 
		# saving the mask image 
		plt.imsave(mask_path, predicted_mask_img) 

		# inpainting 
		strength = float(request.form['strength']) 
		num_inference_steps = int(request.form['num_inference_steps'])  

	inpainted_image = wrinkle_remove_using_stable_diffusion_2_inpaint(img_path, mask_path, strength, num_inference_steps) 
	inpainted_path = "./static/inpainted_img/inpainted.jpg"  #+ img.filename 
	inpainted_image.save(inpainted_path) 


	return render_template("index.html", prediction = "predicted", strength=strength, inference = num_inference_steps)  


if __name__ =='__main__':
	#app.debug = True
	app.run()
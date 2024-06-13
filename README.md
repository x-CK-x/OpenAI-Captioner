# OpenAI-Captioner
Uses GPT models from OpenAI to caption image data in single &amp; batch configurations.

### 3 important steps
1. make an openai account in order to setup an api key (**side note**: openai will mandate that the user sets up a CC with their own respective account to manage the billing for their api access/usage)
2. clone the repo
3. pip install -r requirements.txt

### Usage
There are textbox entries for the user to provide information for the following:
- api key
- prompt
- output PATH
- preset configuration name : for preset dropdown list

#### Additional Features
- image and caption/tag statistics tab, for evaluating as a (post-processing step) all the results from the model
- preset options for the prompt & output path
- new queue system to let the user provide as many images as they want to avoid the 20 image limit with the api calls
- images that might not return captions can temporarily stay on the image upload component, to quickly test other GPT vlm models available to see if those provide captions instead
- a second image upload component was added to let the user add images to the live queue as they deem necessary

## Run Instructions
> python gpt-vlm-run.py

#### Main Menu
![0](https://github.com/x-CK-x/OpenAI-Captioner/assets/48079849/2d6c80a9-119e-4582-a803-38f52a5531ee)


#### Batch Mode Configuration Tab
![1](https://github.com/x-CK-x/OpenAI-Captioner/assets/48079849/1834b23a-a69d-4322-bd86-048da6a6915a)


#### Stats Tab
![2](https://github.com/x-CK-x/OpenAI-Captioner/assets/48079849/957bfcc3-0e2f-4fb1-ac4b-e7031783abd3)


#### Troubleshooting

- if the response from the api call contains something along the lines of: "Sorry image could not be described", then the image is likely to in some way be in violation of OpenAI's T.O.S.
- if the response eludes to a image incompatibility; could be the result of the selected model being deprecated and only raw image urls would be supported (versus the current image string conversion)
- if other errors occur, see: [https://platform.openai.com/docs/guides/error-codes/api-errors.](https://platform.openai.com/docs/guides/error-codes/api-errors.)

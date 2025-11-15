# ComfyUI MiVolo V2 Nodes


[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Original Project](https://img.shields.io/badge/Original%20Model-iitolstykh/mivolo_v2-blue)](https://huggingface.co/iitolstykh/mivolo_v2)

Use the advanced **MiVolo V2** model directly in ComfyUI for high-precision **age and gender estimation**!

This project is a ComfyUI node wrapper for the `iitolstykh/mivolo_v2` model. MiVolo is a powerful Transformer-based, multi-input (face and body) model that provides robust age and gender predictions.

## 🌟 Core Features

* **High-Precision Age Estimation:** Predicts the age of a person from an image.
* **Reliable Gender Prediction:** Predicts the gender (e.g., Male/Female) with a confidence score.
* **Multi-Input Logic:** Leverages the original model's ability to use both face and body crops for more accurate results.
* **Workflow Integration:** Easily chain with face detectors, croppers, and other nodes to analyze images (e.g., AI-generated characters or real photos).

## 🖼️ Nodes & Workflow Examples

This pack includes the following nodes:

### `MiVoloPredictor`
[Screenshot of your custom node's inputs and outputs]

* **Inputs:**
    * `image`: (IMAGE) The input image containing a person.
    * `bbox`: (BBOX) A bounding box (from a detector) for the person. This is highly recommended for best results.
* **Outputs:**
    * `AGE`: (FLOAT) The predicted age.
    * `GENDER`: (STRING) The predicted gender.

### Example Workflow
A visual example of how to connect the nodes.

[Image or GIF of a ComfyUI workflow, e.g., Load Image -> Face Detect -> Crop -> MiVolo Node -> Display Result]

> **Tip:** For the best accuracy, always use a face/person detector (like one from the ImpactPack or Ultralytics nodes) to get a bounding box (`BBOX`) and crop the person *before* passing the image to the MiVolo node. The original model was trained on cropped images.

## 🚀 How to Install

### 1. (Recommended) Using ComfyUI Manager
1.  Open the ComfyUI Manager menu.
2.  Click on "Install Custom Nodes".
3.  Search for `[Your Project Name, e.g., ComfyUI-MiVolo-V2]` and click "Install".
4.  Restart ComfyUI.

### 2. Manual Installation (Git Clone)
1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone [Your GitHub Repository URL]
    ```
3.  Install the required dependencies:
    ```bash
    cd [Your Project Name]
    pip install -r requirements.txt
    ```
    *(Ensure you list all Python packages like `transformers`, `accelerate`, etc., in your `requirements.txt` file)*
4.  Restart ComfyUI.

## 💡 Usage Tips

* The model performs best on clear, front-facing or profile shots, but it is robust.
* You can use the output `AGE` and `GENDER` as inputs for conditioning prompts or for sorting/filtering images.
* Combine with a `Reroute` or `Primitive` node to easily see the text-based gender and float-based age output.

## 📜 Credits & License

This project is an **Adapted Material** based on the work of `iitolstykh/mivolo_v2`.

* **Original Model:** [iitolstykh/mivolo_v2 on Hugging Face](https://huggingface.co/iitolstykh/mivolo_v2)
* **Original Papers:**
    * [MiVOLO: Multi-input Transformer for Age and Gender Estimation (2023)](https://arxiv.org/abs/2307.04616)
    * [Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation (2024)](https://arxiv.org/abs/2403.02302)
* [cite_start]**Original License:** The original project is licensed under a custom "Public license with attribution and conditions reserved" [cite: 1][cite_start], which is a reworking of the CC BY-SA 4.0 license[cite: 18].

[cite_start]As required by the "Conditions Reserved" (ShareAlike) clause of the original license[cite: 66], **this ComfyUI node project is also released under the Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0) license.**

This means you are free to use, modify, and distribute this project and its derivatives, provided you give appropriate attribution and share your adaptations under the same license.

## 🐞 Feedback & Issues

If you encounter any bugs, have feature requests, or want to share a cool workflow, please open an Issue on this repository!
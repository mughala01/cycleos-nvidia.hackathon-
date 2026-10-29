# cycleos-nvidia.hackathon-
# CycleOS

**An AI-powered planning assistant designed around real-life capacity, not perfect productivity.**

## The problem

Overwhelm can make it difficult to decide what to do first, particularly when energy, focus and available time vary from day to day.

## The solution

CycleOS uses an NVIDIA open model through Nebius Token Factory to help users turn their current circumstances into a manageable plan.

Users can provide:
- Energy, focus and mood
- Available time
- Tasks and priorities
- An optional brain dump
- Optional menstrual-cycle context

CycleOS generates practical recommendations based on the information provided. It does not assume that cycle phase determines ability or make medical diagnoses.

## Technology

- Python
- Google Colab
- Nebius Token Factory API
- NVIDIA Nemotron 3 Nano

## Getting started

1. Open `cycleos.ipynb` in Google Colab.
2. Install the required Python package.
3. Enter your Nebius API key when prompted.
4. Run the notebook cells in order.
5. Complete your check-in and receive your plan.

## Privacy and limitations

CycleOS is a planning assistant, not a medical tool. Users should avoid entering sensitive personal information they do not want processed by the model or stored in local history files.

## Future development

- Personalised habit-duration estimates
- Low-, medium- and high-capacity task packages
- User-controlled cycle-context options
- Longer-term learning from user-approved patterns

- ## Try CycleOS

You can try the live version here:

[CycleOS Live Demo](https://cycleos-nvidia-hackathon.onrender.com/)

The app is hosted on Render's free plan, so it might take a minute to load if it hasn't been used recently.

<div align="center">
  
# Computer Vision Analysis of NFL Game Footage
***
![giphy](https://github.com/user-attachments/assets/6106f50a-7eba-4f9a-b369-1534a6b2abf1)
***

### Introduction
This project aims to automate the manual and time-intensive process of breaking down NFL game footage. By leveraging computer vision and machine learning techniques, we can reduce the hours coaches and analysts spend labeling formations, coverages, and player actions.
Automating a part of this project can save a lot of time and labor, allowing coaches to be able to focus on strategy rather than tedious data entry.

### Project Overview

I. Functionalities
| Feature | Use Case |
| ------- | -------- |
| Player Detection & Tracking | - Automatically detect and label each player on the field using bounding boxes <br> - Track players across frames to maintain consistent IDS|
| Play-Level Classification | - Identify offensive formations and defensive startups <br> - Differentiate run vs. pass plays |
| Route & Coverage Recognition | - Classify individual pass routes and coverage types <br> - Provide a summary label for the entire play | 
| Automated Tagging Dashboard | - Present results in a user-friendly interface, where coaches can review and edit any misclassifications quickly | 

#### Data Requirements
I. Video Footage:
**All-22 Film** - overhead angle showing all players

II. Annotations/Labels: 
Manually labeled bounding boxes for a subset of plays; formation & coverage labels for each clip or frame

#### Tech Stack
`Python 3.9+`, `PyTorch/TensorFlow`, `OpenCV`, `YOLO/Detectron2`, `CVAT/Labellmg`

</div>

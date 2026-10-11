# **Industry Project Proposal: PillTally, Counting Pills from a Phone Photo**

Peeranat Patarakittisopol

## **Executive summary**

PillTally counts pills from a single phone photo. A pharmacist spreads the pills on a tray, takes a photo, and gets the count in about a second, with a box drawn around every pill so they can check the result at a glance. This proposal requests one week of engineering time and a small budget to build and test a prototype for the pharmacy team.  
The core model is an object detector (YOLO family) fine-tuned from pretrained weights with transfer learning. The hard part is not 10 pills but 100 to 300 pills that touch each other, look identical and sit on different backgrounds under different light. The model is trained on a mix of real photos and synthetic images made by pasting real pill cut-outs onto tray photos, and it is compared against a classical image-processing counter on a leak-free test set.

## **Problem and business value**

### **The problem**

Pharmacies count pills every day, for example 30, 60 or 90 tablets for a patient on long-term medicine. Most do it by hand on a counting tray, sliding pills five at a time. This is slow, tiring and easy to get wrong when the count is large or the pharmacy is busy. A wrong count means a patient gets too few pills, or the pharmacy loses stock.

### **Why it is not trivial**

Counting 10 well-separated pills is easy for any method, and a person can do it in seconds. It becomes hard when:

> * there are 100 or more pills, many of them touching;  
> * pills are small, round and all identical, so their edges blend together;  
> * the pill color is close to the tray color, or there are shadows and glare;  
> * photos come from different phones, angles and lighting.

Simple image processing (thresholding and splitting blobs) tends to merge touching pills into one blob or split one pill into two. The model must find each pill separately even in a crowded photo.

### **Business value**

> * **Pharmacists:** spend less time on repetitive counting and more on checking prescriptions and advising patients.  
> * **Operations:** fewer miscounts mean fewer patient complaints and less stock loss.

The alternative is a dedicated pill-counting machine, which costs far more and needs space at every counter. PillTally runs on phones the team already uses, so the main cost is engineering time. If the prototype meets the success criteria, the next step is a pilot at one dispensing counter.

## **Approach**

### **Task choice**

Counting is framed as **object detection**: the model draws a box around each pill and the count is the number of boxes. Detection is chosen over simply predicting a number because the boxes let the pharmacist see exactly what was counted and spot mistakes, which matters for trust in a pharmacy.

### **Model**

> * **Detector:** a YOLO-family model, starting from weights pretrained on a large image dataset.  
> * **Transfer learning:** the pretrained model already knows edges, shapes and textures. It is fine-tuned on pill photos so it learns what one pill looks like. With little data, the early layers are frozen at first and unfrozen later at a low learning rate.  
> * **Input size:** photos are processed at a higher resolution than usual (for example 1024 pixels), because pills are small objects in a large image.  
> * **Crowded scenes:** the confidence threshold and the overlap threshold for removing duplicate boxes are tuned on validation data, because crowded, touching pills are where detectors usually miss or double-count.

### **From photo to count**

> 1. The pharmacist spreads one type of pill in a single layer on a tray and takes a photo.  
> 2. The detector finds every pill and returns boxes with confidence scores.  
> 3. The app shows the count and draws the boxes on the photo.  
> 4. Boxes with low confidence are highlighted, so the pharmacist knows where to look before accepting the count.

## **Data**

### **Real photos**

I will collect my own data with 4 to 5 types of over-the-counter pills and vitamins of different shapes, sizes and colors (small white round, large oval, capsule, colored tablet).

> * About 150 photos, from 10 to 300 pills each.  
> * Different backgrounds (white tray, blue tray, dark table), lighting (daylight, indoor light, shadows) and at least two phones.  
> * Pills are spread in one layer, with many touching.  
> * Each photo is taken in a recorded "session" (same pile, same place), which is used for splitting.

### **Labels**

Drawing a box around 300 pills by hand is slow, so labeling is done in two steps: an early model or the classical counter suggests boxes, then I correct them in a labeling tool. For every test photo, the true count is also checked by counting by hand.

### **Synthetic images**

To get many more training images, single pills are cut out from photos and pasted at random positions, angles and densities onto empty tray photos, with random brightness and shadows. Every pasted pill has a known box, so these images are labeled for free. Synthetic images are used for training only, never for testing.

### **Augmentation**

Standard augmentation during training: rotation, flips, scaling, brightness, contrast and blur, so the model handles different phones and lighting.

## **Evaluation plan**

The main claim to test: **the fine-tuned detector counts large, crowded piles much more accurately than classical image processing, on photos taken in conditions it has not seen.**

### **Split and leakage checks**

> * Real photos are split by session (70% train, 15% validation, 15% test), so photos of the same pile never appear in both training and test.  
> * The test set includes at least one background and one lighting condition that do not appear in training, to check how well the model handles new conditions.  
> * Synthetic images are used only in training.  
> * The test set is used once, at the end.

### **Baselines**

| Level | Method | What it tells us |
| :---- | :---- | :---- |
| 1 | Predict the average count of the training set | The floor any method must beat |
| 2 | Classical image processing: threshold, split touching blobs with the watershed method, count blobs | How far a no-learning approach gets |
| 3 | Pretrained detector without fine-tuning | How much fine-tuning on pills adds |
| 5 | Fine-tuned detector (ours) | The full system |

### **Metrics**

> * **Exact count rate:** the share of photos where the count is exactly right. This is the main product metric, because a pharmacy needs the exact number.  
> * **MAE:** on average, how many pills the count is off by.  
> * **Relative error:** error divided by the true count, so a miss of 2 in 300 is judged differently from a miss of 2 in 20\.  
> * **mAP@0.5:** standard detection quality, as a secondary check that boxes are in the right places.

All metrics are reported separately for small (10 to 50), medium (51 to 150\) and large (151 to 300\) piles, because the problem gets harder as the pile grows.

### **Experiments**

| \# | Experiment | What it shows |
| :---- | :---- | :---- |
| 1 | All methods on the test set, by pile size | Where the detector beats classical methods, and by how much |
| 2 | New conditions (unseen background or lighting) | Whether the model works outside the conditions it was trained on |
| 3 | Ablation: without pretraining, without synthetic images, without augmentation | Which part of the training actually helps |
| 4 | Time test: hand counting vs PillTally on the same piles | Whether the tool actually saves time |

### **Confidence and success criteria**

Results get 95% bootstrap confidence intervals over test photos. These criteria are the go/no-go decision for a pilot: an exact count rate of at least 90% on piles up to 150 pills, a clearly lower MAE than the classical method on large piles, and at least 50% less time per count than hand counting, including the pharmacist's check.

## **Scope and timeline**

### **In scope**

> * Counting one type of pill per photo, spread in a single layer  
> * Own dataset of real photos, plus synthetic training images  
> * Classical baseline, pretrained baseline and fine-tuned detector  
> * Experiments 1 to 4  
> * Live demo: take a phone photo of real pills and show the count with boxes

### **Out of scope**

> * Identifying which drug a pill is  
> * Telling apart different pills that look similar in the same photo  
> * Pills stacked on top of each other  
> * A finished mobile app (the demo runs as a simple web page opened on the phone)

### **Timeline**

Everything is due on 10 October.

| Dates | Work | Done when |
| :---- | :---- | :---- |
| 4 Oct | Buy pills, take photos in different conditions, make single-pill cut-outs | Real photo set collected |
| 5 Oct | Label photos, build the synthetic image generator, build the classical baseline | Labeled data and Level 1–2 results |
| 6–7 Oct | Fine-tune the detector, tune thresholds, run ablations | Trained model |
| 8 Oct | Run the test set once, confidence intervals, time test | Results table |
| 9 Oct | Demo page: photo in, count and boxes out | Working demo |
| 10 Oct | Final checks and presentation | Ready to present |

## **Risks and backup plans**

| Risk | Effect | Backup plan |
| :---- | :---- | :---- |
| Labeling takes too long | Not enough real training data | Rely more on synthetic images; use fewer, denser real photos; pre-label with the classical counter |
| Detector misses touching pills in large piles | Low exact count rate on large piles | Higher input size; split the photo into tiles and count each tile; tune the overlap threshold |
| Model fails on new backgrounds or light | Poor real-world use | More varied photos and stronger augmentation; ask users to use a plain tray |
| Synthetic images look too fake | Little gain from them | The ablation shows it; fall back to real photos and augmentation |
| Demo lighting differs from training | Wrong count in the live demo | Bring my own tray and test the room light before presenting |

## **Constraints and request**

> * **Hardware:** runs on phones the team already has; no new devices.  
> * **Budget:** under 1,000 baht for pills and a counting tray; training uses free or existing GPU time.  
> * **Safety:** the pharmacist always confirms the count; every pill is boxed and uncertain ones are highlighted, so the tool assists the check and never replaces it.  
> * **Data privacy:** photos show only pills on a tray, with no patient or prescription information.

Request: one week of engineering time (4 to 10 October) and the budget above. If the prototype meets the success criteria, the next step is a two-week pilot at one dispensing counter, comparing time per count and miscounts with the current process. Later steps are more pill types, telling similar-looking pills apart, and a full mobile app.
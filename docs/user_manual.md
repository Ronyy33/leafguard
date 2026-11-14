# LeafGuard: User Manual

**Rohan Sirsewad (DA24B023)** · DA3408 AI Operations (Prof. Sudarsun) · IIT Madras

*This manual is for farmers, field staff and anyone who wants to check a leaf. No technical knowledge is needed. The same guide is in the app under **Help**.*

## What LeafGuard does
You take a photo of a plant leaf. LeafGuard tells you whether the leaf looks **healthy**, or which **disease** it most likely has, and shows **where on the leaf** it saw the problem.

**Supported plants:**
- **Tomato:** healthy, early blight, late blight, bacterial spot, leaf mold
- **Potato:** healthy, early blight, late blight
- **Apple:** healthy, apple scab

## Opening LeafGuard
Open a web browser (Chrome, Firefox, Edge or Safari) and go to the address your administrator gave you, for example **http://localhost:8080**.

The dot in the top-right corner shows the service status:

| Dot | Meaning |
|---|---|
| 🟢 Green "Online · model v2" | Ready to use |
| 🟠 Orange "Starting up…" | Wait a minute |
| 🔴 Red "Offline" | The service isn't running. Contact your administrator. |

## Checking a leaf

![Check a leaf page](img/ui_index.png)


1. **Take a good photo.**
   - Photograph one leaf, filling most of the frame.
   - Use daylight, and avoid dark shade and strong flash.
   - Use a plain background if you can.
   - Hold still so the photo is sharp.
2. **Choose the photo.** On the *Check a leaf* page, drag the photo onto the dashed box, or click the box and pick the photo. **On a phone, tapping the box opens the camera directly.** A small preview appears.
3. **Click "Check this leaf".** The result appears in about a second.
4. **Read the result:**
   - A ✅ **green box** means the leaf looks healthy.
   - A ⚠️ **red box** shows the most likely disease.
   - **Confidence** shows how sure the system is. Above 90% is very sure.
   - **Other possibilities** lists the next most likely answers.
   - **Where the model looked** colours your photo. Red areas mattered most to the decision.
5. **Tell us if it was right.** Click **Yes, correct**, or click **No, it's wrong** and choose the right answer from the list. This keeps the system accurate.
6. Click **Start over** to check another leaf.

![A result: disease, confidence, other possibilities and the heatmap](img/ui_result.png)

## Messages you might see

| Message | What to do |
|---|---|
| ❓ "This photo doesn't look like a plant leaf" | The photo shows mostly sky, paper, a screen or background. Photograph the leaf up close so it fills the frame. |
| "Photo looks very dark" | Retake it in better light. |
| "Photo looks blurry" | Hold the phone steady and tap the screen to focus on the leaf. |
| "The model is not confident" | Retake the photo, or ask an agronomist. |
| "That file isn't a photo we can read" | Use a JPG, PNG or WEBP photo. iPhone HEIC photos must be converted first. |
| "That photo is too large" | The limit is 5 MB. Choose a smaller photo size in your camera settings. |
| "Cannot reach the LeafGuard service" | The service is offline. Try later or contact your administrator. |

## Important
LeafGuard gives **guidance only**. Before costly actions such as spraying or removing plants, confirm with an agricultural expert.

---

## For operators

**Starting the system** (needs Docker):
```bash
docker compose up -d --build     # first start takes a while (downloads)
docker compose ps                # every service should be "running"/"healthy"
```

**Addresses:**

| What | Address |
|---|---|
| App | http://localhost:8080 |
| Pipeline console | http://localhost:8080/pipeline.html |
| Grafana dashboards | http://localhost:3000 (admin / admin) |
| MLflow | http://localhost:5000 |
| API documentation | http://localhost:8000/docs |

**Pipeline console:**

| Area | What it shows or does |
|---|---|
| Pipeline DAG | The data and training steps and their times from the last run. |
| Data drift | Whether recent photos differ from the training photos. Sustained drift retrains automatically. |
| Model registry | All model versions. The one marked *production* is live. |
| **Run pipeline now** | Retrains from scratch. The new model goes live only if it is at least as accurate. |
| **Roll back model** | Returns to the previous version. Click twice to confirm. |
| Live monitoring | Traffic, response times, errors and drift. |

![Pipeline console](img/ui_pipeline.png)

**Stopping the system:** `docker compose down`. Your data and models are kept.

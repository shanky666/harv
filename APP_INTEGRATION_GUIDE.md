# HarvestLenz — Mobile & Web App Integration Guide

This guide provides complete instructions and API references for integrating the **HarvestLenz** fruit quality analysis, fruit verification, and AI grading backend into mobile applications (Flutter, React Native, iOS/Swift, Android/Kotlin) and frontend web clients.

---

## 1. Overview & Setup

* **Backend Base URL**: `http://<server-ip>:8001` (Local dev: `http://127.0.0.1:8001`)
* **Interactive API Docs (Swagger)**: `http://<server-ip>:8001/docs`
* **Supported Content Types**: `multipart/form-data`, `application/json`
* **Authentication**: JWT Bearer Token (`Authorization: Bearer <access_token>`)

---

## 2. Authentication Flow

### 2.1 Register User
* **Endpoint**: `POST /auth/register`
* **Content-Type**: `application/json`

**Request Body**:
```json
{
  "name": "Alex Farmer",
  "email": "alex@harvestlenz.com",
  "phone": "+919876543210",
  "password": "SecurePassword123",
  "location": "Maharashtra, India"
}
```

### 2.2 Login & Obtain JWT Token
* **Endpoint**: `POST /auth/login`
* **Content-Type**: `application/json`

**Request Body**:
```json
{
  "email": "alex@harvestlenz.com",
  "password": "SecurePassword123"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1Ni...",
  "refresh_token": "eyJhbGciOiJIUzI1Ni...",
  "token_type": "bearer"
}
```

> **Header for Authenticated Calls**:
> `Authorization: Bearer eyJhbGciOiJIUzI1Ni...`

---

## 3. Supported Items Endpoint

Retrieve the full list of supported fruits and vegetables for dropdown menus:

* **Endpoint**: `GET /fruits/supported`

**Response**:
```json
{
  "fruits": [
    { "key": "mango", "name": "Mango", "scientific": "Mangifera indica" },
    { "key": "pineapple", "name": "Pineapple", "scientific": "Ananas comosus" },
    { "key": "grapes", "name": "Grapes", "scientific": "Vitis vinifera" },
    { "key": "pomegranate", "name": "Pomegranate", "scientific": "Punica granatum" },
    { "key": "orange", "name": "Orange", "scientific": "Citrus sinensis" },
    { "key": "guava", "name": "Guava", "scientific": "Psidium guajava" },
    { "key": "kiwi", "name": "Kiwi", "scientific": "Actinidia deliciosa" },
    { "key": "watermelon", "name": "Watermelon", "scientific": "Citrullus lanatus" },
    { "key": "banana", "name": "Banana", "scientific": "Musa acuminata" },
    { "key": "cocoa", "name": "Cocoa", "scientific": "Theobroma cacao" },
    { "key": "coffee", "name": "Coffee", "scientific": "Coffea arabica" },
    { "key": "strawberry", "name": "Strawberry", "scientific": "Fragaria x ananassa" },
    { "key": "plum", "name": "Plum", "scientific": "Prunus domestica" },
    { "key": "peach", "name": "Peach", "scientific": "Prunus persica" },
    { "key": "pear", "name": "Pear", "scientific": "Pyrus communis" }
  ],
  "vegetables": [
    { "key": "carrot", "name": "Carrot", "scientific": "Daucus carota" },
    { "key": "tomato", "name": "Tomato", "scientific": "Solanum lycopersicum" },
    { "key": "onion", "name": "Onion", "scientific": "Allium cepa" },
    { "key": "cucumber", "name": "Cucumber", "scientific": "Cucumis sativus" },
    { "key": "capsicum", "name": "Capsicum", "scientific": "Capsicum annuum" },
    { "key": "potato", "name": "Potato", "scientific": "Solanum tuberosum" }
  ]
}
```

---

## 4. Main App Workflow: Fruit Quality Analysis

The quality analysis pipeline follows an async job workflow: **Upload -> Poll Status -> Fetch Full Results**.

```
[Mobile/Web App]
       │
       ├─► 1. POST /analyze?fruit_type=mango (Upload Image)
       │      └─► Returns {"session_id": "...", "status": "processing"}
       │
       ├─► 2. GET /analysis/{session_id}/status (Poll every 1.5s)
       │      └─► Returns {"status": "complete"}
       │
       └─► 3. GET /analysis/{session_id} (Fetch Report)
              └─► Displays Quality Grade, Shelf Life, Market Price & Mismatch Warnings
```

### Step 1: Upload Image for Analysis
* **Endpoint**: `POST /analyze`
* **Query Parameters**:
  * `fruit_type` *(string, required)*: Selected fruit/veg key (e.g. `mango`, `pineapple`, `grapes`) or `"auto"` for automatic AI fruit classification!
  * `is_single` *(boolean, optional, default: `true`)*: `true` for single item, `false` for multi-item basket.
* **Content-Type**: `multipart/form-data`
* **Body Form Field**: `file` (Binary image file: JPG, PNG, WEBP)

**Sample Curl**:
```bash
curl -X POST "http://127.0.0.1:8001/analyze?fruit_type=mango&is_single=true" \
  -H "Authorization: Bearer <token>" \
  -F "file=@fruit_photo.jpg"
```

**Response**:
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "status": "processing",
  "fruit_type": "mango"
}
```

### Step 2: Poll Analysis Job Status
* **Endpoint**: `GET /analysis/{session_id}/status`

**Response (Processing)**:
```json
{
  "status": "processing",
  "error": null
}
```

**Response (Complete)**:
```json
{
  "status": "complete",
  "error": null
}
```

### Step 3: Retrieve Complete Analysis & Quality Report
* **Endpoint**: `GET /analysis/{session_id}`

**Response (Valid Match)**:
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "total_fruits": 1,
  "fruit_type": "mango",
  "overall_grade": "Premium",
  "score": 95,
  "average_shelf_life": 7.0,
  "total_price": 52.50,
  "estimated_selling_price": 68.25,
  "profit_estimation": 15.75,
  "recommended_market": "Export",
  "fruits": [
    {
      "fruit_id": "FRUIT_0001",
      "fruit_type": "mango",
      "grade": "Good",
      "confidence": 0.9542,
      "quality_confidence": 0.9542,
      "bbox": [24, 18, 200, 195],
      "crop_path": "storage/crops/a1b2c3d4.../FRUIT_0001.jpg",
      "shelf_life_days": 7,
      "shelf_life": "7 days",
      "market_recommendation": "Export",
      "price": 52.50,
      "defect_score": 0.0210,
      "defects": [],
      "grade_color": "#66BB6A"
    }
  ],
  "summary": { "good": 1, "better": 0, "reject": 0 },
  "ai_recommendations": [
    "Premium quality Mango detected. Suitable for export or high-end retail."
  ]
}
```

**Response (Wrong Fruit Upload / Mismatch Detected)**:
If a wrong fruit photo was uploaded (e.g. uploaded a Strawberry while *Mango* was selected in dropdown):
```json
{
  "session_id": "b9876543-21fe-4321-dcba-0987654321ba",
  "total_fruits": 1,
  "fruit_type": "mango",
  "overall_grade": "Mismatch",
  "score": 0,
  "average_shelf_life": 0.0,
  "total_price": 0.0,
  "estimated_selling_price": 0.0,
  "profit_estimation": 0.0,
  "recommended_market": "Processing Industry",
  "fruits": [
    {
      "fruit_id": "FRUIT_0001",
      "fruit_type": "mango",
      "grade": "Mismatch",
      "confidence": 1.0,
      "price": 0.0,
      "shelf_life_days": 0,
      "shelf_life": "0 days (Mismatch)",
      "market_recommendation": "N/A - Fruit Mismatch",
      "grade_color": "#E53935"
    }
  ],
  "ai_recommendations": [
    "Image Mismatch: Uploaded image appears to be a 'Strawberry', but 'Mango' was selected in the dropdown."
  ]
}
```

---

## 5. Single Fruit Classifier Endpoint (Instant Debug Mode)

Classify an image crop directly without queuing a background job:

* **Endpoint**: `POST /scan/classify`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `file`

**Response**:
```json
{
  "filename": "strawberry_crop.jpg",
  "fruit_type": "strawberry",
  "confidence": 0.85,
  "classifier_status": "loaded",
  "classifier_classes": ["banana", "grapes", "guava", "mango", "orange", "pineapple", "pomegranate", "strawberry"]
}
```

---

## 6. Model Weights & Classification Setup

To enable deep learning classification (e.g. Fruits-360 / MobileNetV2):

1. **Fruit Type Classifier**:
   Place model weights and class index mapping in:
   * `backend/backend/models/fruit_classifier.h5`
   * `backend/backend/models/class_indices.json`

2. **Per-Fruit Quality Grading Weights**:
   Place fine-tuned grading `.keras` or `.h5` files in:
   * `backend/backend/app/models/weights/mango.keras`
   * `backend/backend/app/models/weights/pineapple.keras`
   * `backend/backend/app/models/weights/grapes.keras`
   * `backend/backend/app/models/weights/pomegranate.keras`
   * `backend/backend/app/models/weights/orange.keras`
   * `backend/backend/app/models/weights/guava.keras`
   * `backend/backend/app/models/weights/strawberry.keras`
   * `backend/backend/app/models/weights/banana.h5`

---

## 7. Sample Client Implementations

### Flutter (Dart)
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

Future<Map<String, dynamic>> analyzeFruit(File imageFile, String fruitType) async {
  final uri = Uri.parse('http://192.168.1.100:8001/analyze?fruit_type=$fruitType&is_single=true');
  var request = http.MultipartRequest('POST', uri);
  request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

  var streamedResponse = await request.send();
  var response = await http.Response.fromStream(streamedResponse);
  var data = jsonDecode(response.body);
  String sessionId = data['session_id'];

  // Poll until complete
  while (true) {
    await Future.delayed(Duration(milliseconds: 1500));
    var statusRes = await http.get(Uri.parse('http://192.168.1.100:8001/analysis/$sessionId/status'));
    var statusData = jsonDecode(statusRes.body);
    if (statusData['status'] == 'complete') break;
  }

  // Get full report
  var reportRes = await http.get(Uri.parse('http://192.168.1.100:8001/analysis/$sessionId'));
  return jsonDecode(reportRes.body);
}
```

### React Native / JavaScript
```javascript
async function analyzeFruit(imageUri, fruitType) {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'photo.jpg',
  });

  const uploadRes = await fetch(`http://192.168.1.100:8001/analyze?fruit_type=${fruitType}&is_single=true`, {
    method: 'POST',
    body: formData,
  });
  const { session_id } = await uploadRes.json();

  // Poll job status
  while (true) {
    await new Promise(r => setTimeout(r, 1500));
    const statusRes = await fetch(`http://192.168.1.100:8001/analysis/${session_id}/status`);
    const status = await statusRes.json();
    if (status.status === 'complete') break;
  }

  // Fetch final analysis
  const finalRes = await fetch(`http://192.168.1.100:8001/analysis/${session_id}`);
  return await finalRes.json();
}
```

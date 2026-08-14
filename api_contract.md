# HarvestLenz API Contract

This document outlines the API endpoints, request schemas, and response formats for the HarvestLenz AI-Powered Fruit Quality Intelligence system, enabling clean end-to-end integration between the Web Frontend mobile application and the FastAPI/AI backend.

---

## Base URL
* **Development/Local**: `http://localhost:8000`
* **Production**: `http://<production-domain>`

---

## Authentication
All endpoints under `/scan` and `/users` require JWT Bearer Authentication. 
Include the JWT token in the request headers:
```http
Authorization: Bearer <access_token>
```

---

## Endpoints

### 1. Register Account
* **Endpoint**: `POST /auth/register`
* **Summary**: Registers a new farmer profile.
* **Authentication**: None
* **Request Body** (`application/json`):
  ```json
  {
    "name": "John Doe",
    "email": "farmer.john@example.com",
    "phone": "+919876543210",
    "password": "securepassword123",
    "location": "Nashik, Maharashtra"
  }
  ```
* **Success Response** (`201 Created`):
  ```json
  {
    "id": "b1a457df-4f27-46c9-a9a3-c5b6510f6a27",
    "name": "John Doe",
    "email": "farmer.john@example.com",
    "phone": "+919876543210",
    "location": "Nashik, Maharashtra",
    "created_at": "2026-06-16T01:42:32Z"
  }
  ```
* **Error Response** (`400 Bad Request`):
  ```json
  {
    "detail": "Email already registered"
  }
  ```

### 2. Login
* **Endpoint**: `POST /auth/login`
* **Summary**: Verifies credentials and issues access/refresh tokens.
* **Authentication**: None
* **Request Body** (`application/json`):
  ```json
  {
    "email": "farmer.john@example.com",
    "password": "securepassword123"
  }
  ```
* **Success Response** (`200 OK`):
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer"
  }
  ```
* **Error Response** (`401 Unauthorized`):
  ```json
  {
    "detail": "Invalid credentials"
  }
  ```

### 3. Get Current User Info
* **Endpoint**: `GET /auth/me`
* **Summary**: Retrieves credentials/claims decoded from JWT.
* **Authentication**: Bearer Token
* **Success Response** (`200 OK`):
  ```json
  {
    "id": "b1a457df-4f27-46c9-a9a3-c5b6510f6a27",
    "name": "John Doe",
    "email": "farmer.john@example.com",
    "phone": "+919876543210",
    "location": "Nashik, Maharashtra",
    "created_at": "2026-06-16T01:42:32Z"
  }
  ```

### 4. Upload Basket Image
* **Endpoint**: `POST /scan/upload`
* **Summary**: Uploads a single raw basket image. Stores the original, generates a database scan record, and returns metadata.
* **Authentication**: Bearer Token
* **Request Body** (`multipart/form-data`):
  * `file`: (Binary image file, JPG/PNG/WEBP)
* **Success Response** (`201 Created`):
  ```json
  {
    "scan_id": "c8b4f0b2-ae31-4dbb-871d-15ba17ff5bc3",
    "user_id": "b1a457df-4f27-46c9-a9a3-c5b6510f6a27",
    "image_path": "storage/uploads/original/c8b4f0b2-ae31-4dbb-871d-15ba17ff5bc3.jpg",
    "total_fruits": 0,
    "scan_date": "2026-06-16T01:43:00Z"
  }
  ```
* **Error Response** (`400 Bad Request`):
  ```json
  {
    "detail": "Unsupported file type: image/gif"
  }
  ```

### 5. Process Crate/Basket Scan
* **Endpoint**: `POST /scan/process/{scan_id}`
* **Summary**: Processes the uploaded image. Runs YOLOv8 multi-fruit detection, crop extraction, CNN classification & grading, post-harvest shelf life & market recommendation pipeline, generates passports and compiling report.
* **Authentication**: Bearer Token
* **Success Response** (`200 OK`):
  ```json
  {
    "scan_id": "c8b4f0b2-ae31-4dbb-871d-15ba17ff5bc3",
    "total_fruits": 18,
    "good": 10,
    "better": 5,
    "medium": 2,
    "reject": 1,
    "average_shelf_life": "6.8 days",
    "fruit_distribution": {
      "Mango": 10,
      "Orange": 5,
      "Grapes": 3
    },
    "markets": {
      "best_market": "Local Mandi",
      "expected_price": "₹1200 - ₹1500 / basket",
      "alternatives": [
        {
          "market": "Export Center",
          "price": "₹2000 - ₹2400 / basket"
        }
      ]
    }
  }
  ```

### 6. Get Quality Report
* **Endpoint**: `GET /scan/report/{scan_id}`
* **Summary**: Returns quality aggregation stats and the PDF download URL.
* **Authentication**: Bearer Token
* **Success Response** (`200 OK`):
  ```json
  {
    "scan_id": "c8b4f0b2-ae31-4dbb-871d-15ba17ff5bc3",
    "total_fruits": 18,
    "grades": {
      "good": 10,
      "better": 5,
      "medium": 2,
      "reject": 1
    },
    "shelf_life": "6.8 days",
    "market": "Local Mandi",
    "pdf_url": "/scan/report/c8b4f0b2-ae31-4dbb-871d-15ba17ff5bc3/pdf",
    "created_at": "2026-06-16T01:44:12Z"
  }
  ```

### 7. Download PDF Report
* **Endpoint**: `GET /scan/report/{scan_id}/pdf`
* **Summary**: Stream downloads the compiled PDF report file.
* **Authentication**: Bearer Token
* **Success Response** (`200 OK` / `application/pdf`):
  * Raw PDF file stream.

### 8. Get AI Fruit Passport
* **Endpoint**: `GET /scan/passport/{fruit_id}`
* **Summary**: Retrieves the scientific facts, grade, detected defects, and market guidelines for a single detected fruit crop.
* **Authentication**: Bearer Token
* **Success Response** (`200 OK`):
  ```json
  {
    "passport_id": "df5e8c1b-e538-4b77-9ff5-7aa8e7ff2b38",
    "fruit_id": "e9b21a5c-7d9a-4c28-912b-312ab62153cb",
    "fruit_type": "Mango",
    "grade": "Good",
    "defects": "Mild anthracnose spot",
    "shelf_life": "8 days",
    "market": "Export Hub",
    "created_at": "2026-06-16T01:44:15Z"
  }
  ```

### 9. Get User Profile
* **Endpoint**: `GET /users/profile`
* **Summary**: Retrieves the authenticated user profile information.
* **Authentication**: Bearer Token
* **Success Response** (`200 OK`):
  ```json
  {
    "id": "b1a457df-4f27-46c9-a9a3-c5b6510f6a27",
    "name": "John Doe",
    "email": "farmer.john@example.com",
    "phone": "+919876543210",
    "location": "Nashik, Maharashtra",
    "created_at": "2026-06-16T01:42:32Z"
  }
  ```

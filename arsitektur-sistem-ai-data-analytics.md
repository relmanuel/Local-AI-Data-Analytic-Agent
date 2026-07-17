# Arsitektur Sistem: AI Data Analytics Offline & Lokal

Sistem ini dirancang untuk bekerja 100% offline demi menjaga privasi data, dapat terhubung ke database live secara aman, dan dijalankan menggunakan bahasa pemrograman Python.

## 1. Diagram Arsitektur Blok

```
+------------------------------------------------------------------------+
|                           USER INTERFACE                               |
|  [ Streamlit Web App ]                                                 |
|  - Menerima pertanyaan natural (contoh: "Tren penjualan bulan lalu")   |
|  - Menampilkan jawaban teks, tabel data, dan grafik interaktif         |
+-----------------------------------^------------------------------------+
                                    |
                        User Query  |  Data / Charts
                                    v
+------------------------------------------------------------------------+
|                        AI AGENT FRAMEWORK                              |
|  [ Pydantic AI / LangChain ]                                           |
|  - Mengelola memori percakapan dan alur instruksi (Prompt Routing)     |
|  - Memilih 'Tools' (Python/SQL) yang sesuai dengan kebutuhan pengguna  |
+-------------------+--------------------------------^-------------------+
                    |                                |
      User Prompt   |  Model Output                  |  Execution Results
      + Context     |  (Code / Query)                |  (Dataframes/Plots)
                    v                                |
+-----------------------+   +------------------------+-------------------+
|     LOCAL LLM         |   |             LOCAL EXECUTION ENGINE         |
|                       |   |                                            |
|  [ Ollama / Jan.ai ]  |   |  +--------------------------------------+  |
|  - Llama-3 / Qwen-2.5 |   |  |        SECURE CODE SANDBOX           |  |
|  - Eksekusi Offline   |   |  |  - Memuat data ke Pandas DataFrame   |  |
|  - Otomatisasi kode   |   |  |  - Membuat grafik Plotly/Matplotlib  |  |
|    Python atau SQL    |   |  +------------------^-------------------+  |
+-----------------------+   +---------------------|----------------------+
                                                  |
                                       SQL / Read | Data
                                       Commands   | Stream
                                                  v
+------------------------------------------------------------------------+
|                             DATA LAYER                                 |
|                                                                        |
|  [ Static Files ]                      [ Live Databases ]              |
|  - CSV, Excel, Parquet                 - PostgreSQL, MySQL, SQLite     |
+------------------------------------------------------------------------+
```

## 2. Penjelasan Aliran Data (Data Flow)

1. **Input Pengguna**: Anda mengetik pertanyaan kasual di browser lokal melalui antarmuka Streamlit.
2. **Perencanaan Agen**: Pydantic AI menerima teks tersebut, menambahkan konteks (seperti struktur tabel), lalu mengirimkannya ke Local LLM.
3. **Generasi Kode**: LLM yang berjalan di dalam Ollama (tanpa internet) menganalisis pertanyaan dan menulis kode Python atau query SQL yang dibutuhkan.
4. **Eksekusi Aman**: Agen mengambil kode tersebut dan menjalankannya di dalam Local Execution Engine. Mesin ini langsung menarik data dari Database Live atau File Statis.
5. **Visualisasi & Jawaban**: Hasil olahan data (tabel Pandas atau grafik Plotly) dikirim kembali ke UI Streamlit untuk ditampilkan kepada Anda.

## 3. Komponen Utama Arsitektur

- **Front-End (Streamlit)**: Bertindak sebagai dashboard interaktif ringan yang berjalan di browser lokal (localhost).
- **Brain Engine (Ollama)**: Menjalankan model pintar berukuran kecil yang dioptimalkan untuk coding (seperti Qwen-2.5-Coder atau Llama-3).
- **Security Barrier (Sandbox Execution)**: Memastikan kode Python/SQL yang dibuat oleh AI hanya memiliki akses membaca (Read-Only), sehingga AI tidak bisa menghapus atau merusak data penting Anda.
- **Data Connector**: Lapisan penghubung menggunakan SQLAlchemy atau Pandas untuk membaca database lokal maupun file mentah secara instan.

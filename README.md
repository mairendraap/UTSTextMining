# UTS Text Mining — Text Processing Bahasa Indonesia

Demo end-to-end preprocessing teks Bahasa Indonesia + TF-IDF.

## Struktur

```
├── UTS_TextProcessing.ipynb   # notebook utama (run dari atas ke bawah)
├── dataset/
│   ├── dataset_dummy_ulasan.csv  # input mentah (30 baris, buatan sendiri)
│   └── ulasan_bersih.csv         # output (dibuat otomatis cell terakhir)
└── docs/
    └── script_penjelasan.md      # naskah presentasi per cell + kode
```

## Dataset

`dataset/dataset_dummy_ulasan.csv` — 30 ulasan produk Bahasa Indonesia.

| kolom | arti |
|---|---|
| `id` | nomor unik |
| `ulasan` | teks mentah |
| `rating` | 1–5 |
| `label` | positif / netral / negatif (12 / 7 / 11) |

Sengaja berisik (kapital, tanda seru, singkatan) agar tahap cleaning terlihat.

## Cara jalan

```bash
pip install pandas scikit-learn matplotlib Sastrawi jupyter
jupyter notebook UTS_TextProcessing.ipynb
# lalu: Run All
```

Tanpa `Sastrawi` pun tetap jalan (ada fallback stemming sederhana).

## Alur

1. Cleaning + case folding (regex)
2. Tokenizing (`split`)
3. Stopword removal (list manual 40 kata ID)
4. Stemming (Sastrawi Nazief-Adriani + fallback)
5. TF-IDF (`max_features=100`) + plot Top-15 kata
6. Simpan `dataset/ulasan_bersih.csv`

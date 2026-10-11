# Naskah Penjelasan — UTS Text Mining: Text Processing

> Cara pakai naskah ini: bacakan per bagian sambil **Run cell yang sama** di `UTS_TextProcessing.ipynb`.
> Setiap bagian pola-nya: **(1) Yang dibacakan → (2) Kode yang di-run → (3) Yang ditunjuk di output.**
> Estimasi total: 7–10 menit.

---

## 0. Pembuka (30 detik)

> "Assalamualaikum / selamat pagi. Saya akan mendemokan alur **Text Processing Bahasa Indonesia** untuk UTS Text Mining.
> Dataset-nya `dataset/dataset_dummy_ulasan.csv` — 30 ulasan produk buatan sendiri dengan kolom `id, ulasan, rating, label`.
> Alurnya: **Loading → Cleaning + Case Folding → Tokenizing → Stopword Removal → Stemming → TF-IDF → Visualisasi.**
> Semua kode ada di `UTS_TextProcessing.ipynb`. Saya run dari atas ke bawah."

**Run:** belum run apa-apa, tunjukkan struktur file:

```
UTSTextMining/
├── UTS_TextProcessing.ipynb
├── dataset/
│   ├── dataset_dummy_ulasan.csv   <- input mentah
│   └── ulasan_bersih.csv          <- output (dibuat cell terakhir)
└── docs/
    └── script_penjelasan.md       <- file ini
```

---

## 1. Setup Library (30 detik)

**Bacakan:**

> "Cell pertama hanya import. `pandas` untuk tabel, `re` untuk regex cleaning, `sklearn TfidfVectorizer` untuk pembobotan kata, `matplotlib` untuk grafik.
> Untuk stemming Bahasa Indonesia saya pakai **Sastrawi**. Kalau belum terinstall, notebook tetap jalan pakai fungsi fallback sederhana — jadi tidak error saat demo."

**Kode yang di-run (Cell 1):**

```python
import re
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Stemming Sastrawi (opsional, ada fallback)
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    print("Sastrawi OK")
except ImportError:
    stemmer = None
    print("Sastrawi tidak terinstall → pakai fallback. Install via: pip install Sastrawi")
```

**Tunjuk output:** `Sastrawi OK` atau pesan fallback. Jelaskan: "Kalau muncul fallback pun tidak masalah."

---

## 2. Dataset yang Digunakan (1 menit)

**Bacakan:**

> "Dataset-nya dummy, saya buat sendiri 30 baris agar seimbang dan berisik seperti data asli.
> Kolomnya ada empat: `id` sebagai nomor unik, `ulasan` sebagai teks mentah, `rating` 1 sampai 5, dan `label` sentimen — positif, netral, negatif.
> Komposisinya 12 positif, 11 negatif, 7 netral. Sengaja ada huruf kapital, tanda seru, singkatan — supaya tahap cleaning kelihatan gunanya."

**Kode yang di-run (Cell 2):**

```python
df = pd.read_csv("dataset/dataset_dummy_ulasan.csv").copy()  # .copy(): pastikan df pemilik data (aman untuk pandas 3 Copy-on-Write)
print("Shape:", df.shape)
print(df["label"].value_counts())
print(df.head(5).to_string())
```

**Tunjuk output:** `(30, 4)`, distribusi label, dan 5 contoh ulasan. Contoh kalimat yang bagus ditunjuk:

- Positif: `"Produk bagus banget, pengiriman cepat dan packing rapi!"`
- Negatif: `"Kecewa, barang datang rusak dan pelayanan lambat."`
- Netral: `"Kualitas lumayan, harga sesuai tapi pengiriman agak lama."`

---

## 3. Cleaning + Case Folding (1 menit)

**Bacakan:**

> "Teks mentah tidak bisa langsung dipakai model. Ada URL, mention, angka, tanda baca.
> Fungsi `clean_text` melakukan: lowercasing dulu, lalu hapus URL, mention, hashtag dengan regex, sisakan huruf a-z, lalu rapikan spasi.
> Contohnya kalimat pertama: huruf kapital hilang, tanda seru hilang, hasilnya bersih."

**Kode yang di-run (Cell 3):**

```python
def clean_text(teks: str) -> str:
    teks = teks.lower()                       # case folding
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)  # hapus URL
    teks = re.sub(r"@[A-Za-z0-9_]+", " ", teks)      # hapus mention
    teks = re.sub(r"#[A-Za-z0-9_]+", " ", teks)      # hapus hashtag
    teks = re.sub(r"[^a-z\s]", " ", teks)            # sisakan huruf a-z + spasi
    teks = re.sub(r"\s+", " ", teks).strip()        # rapikan spasi
    return teks

df.loc[:, "clean"] = df["ulasan"].astype(str).apply(clean_text)  # .loc[:, kol]: assignment satu langkah (anti ChainedAssignmentError pandas 3)
print("SEBELUM:", df.loc[0, "ulasan"])
print("SESUDAH:", df.loc[0, "clean"])
```

**Tunjuk output:**

```
SEBELUM: Produk bagus banget, pengiriman cepat dan packing rapi!
SESUDAH: produk bagus banget pengiriman cepat dan packing rapi
```

---

## 4. Tokenizing (30 detik)

**Bacakan:**

> "Setelah bersih, kalimat dipecah jadi list kata — namanya token.
> Karena teks sudah bersih, cukup pakai `split()`. Hasilnya bisa dihitung frekuensinya."

**Kode yang di-run (Cell 4):**

```python
df.loc[:, "tokens"] = df["clean"].str.split()
print(df.loc[0, "tokens"])
print("Jumlah token kalimat-1:", len(df.loc[0, "tokens"]))
```

**Tunjuk output:** `['produk', 'bagus', 'banget', 'pengiriman', 'cepat', 'dan', 'packing', 'rapi']` — 8 token.

---

## 5. Stopword Removal (1 menit)

**Bacakan:**

> "Tidak semua kata berguna. Kata seperti `dan, yang, sangat, banget` muncul di mana-mana tapi tidak membedakan sentimen positif vs negatif.
> Saya pakai list manual 40 stopword Indonesia agar notebook jalan tanpa download NLTK.
> Perhatikan kata `dan` dan `banget` hilang setelah filtering."

**Kode yang di-run (Cell 5):**

```python
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan",
    "tidak", "tak", "bukan", "sangat", "sekali", "banget", "juga",
    "sudah", "telah", "ada", "adalah", "atau", "tapi", "namun",
    "karena", "jika", "kalau", "saat", "ketika", "saya", "kamu",
    "dia", "mereka", "kami", "kita", "nya", "lah", "kok", "sih",
    "aja", "saja", "pada", "dalam", "secara", "sering", "selalu"
}

df.loc[:, "no_stop"] = df["tokens"].apply(lambda toks: [t for t in toks if t not in STOPWORDS_ID])
print("SEBELUM:", df.loc[0, "tokens"])
print("SESUDAH:", df.loc[0, "no_stop"])
```

**Tunjuk output:** dari 8 token → 6 token (`dan`, `banget` terbuang).

---

## 6. Stemming (1 menit)

**Bacakan:**

> "Bahasa Indonesia kaya imbuhan: `pengiriman, dikirim, mengirim` itu satu akar kata `kirim`. Kalau tidak disamakan, model menganggapnya tiga fitur berbeda.
> Saya pakai **Sastrawi dengan algoritma Nazief-Adriani** khusus Bahasa Indonesia. Kalau Sastrawi belum install, ada fallback penghapus awalan-akhiran sederhana.
> Contohnya `pengiriman` menjadi `kirim`. Lalu token digabung lagi jadi kalimat bersih di kolom `final`."

**Kode yang di-run (Cell 6):**

```python
def simple_stem_fallback(kata: str) -> str:
    """Fallback konservatif (aman, tidak merusak kata).
    Hanya hapus awalan/akhiran yang aman. Untuk imbuhan nasal
    (meng-/peng-/meny-/peny-) WAJIB pakai Sastrawi agar benar.
    Contoh: pengiriman -> kirim (Sastrawi), bukan irim (fallback kasar).
    """
    for pre in ("ber", "ter", "di", "ke", "se"):
        if kata.startswith(pre) and len(kata) > len(pre) + 3:
            kata = kata[len(pre):]
            break
    for suf in ("kan", "an", "nya", "lah", "kah", "i"):
        if kata.endswith(suf) and len(kata) > len(suf) + 3:
            kata = kata[:-len(suf)]
            break
    return kata

def stem_tokens(tokens):
    if stemmer is not None:
        return [stemmer.stem(t) for t in tokens]
    return [simple_stem_fallback(t) for t in tokens]

df.loc[:, "stemmed"] = df["no_stop"].apply(stem_tokens)
df.loc[:, "final"] = df["stemmed"].apply(lambda toks: " ".join(toks))
print("SEBELUM:", df.loc[0, "no_stop"])
print("SESUDAH:", df.loc[0, "stemmed"])
print("FINAL  :", df.loc[0, "final"])
```

**Tunjuk output:** `pengiriman → kirim`. Kalimat final mis. `"produk bagus kirim cepat packing rapi"`.

---

## 7. TF-IDF + Visualisasi (1,5 menit)

**Bacakan:**

> "Model ML tidak paham teks, hanya paham angka. Jadi kalimat bersih diubah jadi vektor angka pakai **TF-IDF**.
> TF mengukur seberapa sering kata muncul di satu dokumen, IDF menghukum kata yang muncul di semua dokumen seperti kata `produk` — karena tidak pembeda.
> Hasilnya matriks 30 baris kali maksimal 100 kolom fitur — siap masuk Naive Bayes, SVM, atau Logistic Regression.
> Sebagai cek sanity, saya plot 15 kata tersering setelah preprocessing. Kalau kata sampah masih dominan, berarti stopword kurang."

**Kode yang di-run (Cell 7):**

```python
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df["final"])
print("Bentuk matriks TF-IDF:", X.shape)  # (30 dokumen, <=100 fitur)
print("Contoh fitur:", list(vectorizer.get_feature_names_out()[:10]))

# Visualisasi frekuensi
freq = Counter(w for toks in df["stemmed"] for w in toks)
top15 = freq.most_common(15)
kata, hitung = zip(*top15)

plt.figure()
plt.barh(list(kata)[::-1], list(hitung)[::-1])
plt.xlabel("Frekuensi")
plt.title("Top 15 Kata Setelah Preprocessing")
plt.tight_layout()
plt.show()

print("Top 15:", top15)
```

**Tunjuk output:** shape `(30, <=100)`, grafik bar horizontal. Biasanya kata top: `sesuai, kualitas, kirim, bagus, cepat`.

---

## 8. Simpan & Penutup (30 detik)

**Bacakan:**

> "Terakhir, hasil bersih disimpan ke `dataset/ulasan_bersih.csv` dengan kolom `id, ulasan, clean, final, label` — file ini yang dipakai untuk training klasifikasi sentimen.
> Kesimpulannya: urutan wajib **cleaning → tokenizing → stopword → stemming → TF-IDF**, tidak boleh dibalik.
> Next step UTS: split train-test, klasifikasi, evaluasi accuracy dan F1. Terima kasih."

**Kode yang di-run (Cell 8):**

```python
df[["id", "ulasan", "clean", "final", "label"]].to_csv("dataset/ulasan_bersih.csv", index=False)
print("Tersimpan: dataset/ulasan_bersih.csv")
print(df[["ulasan", "final", "label"]].head(10).to_string())
```

---

## Lampiran: Pertanyaan Dosen yang Sering Muncul

1. **Kenapa tidak pakai NLTK word_tokenize?** — Karena teks sudah bersih (huruf+spasi), `split()` cukup dan tanpa dependensi download `punkt`.
2. **Kenapa stopword manual?** — Agar reproducible offline. Untuk produksi bisa ganti NLTK/Sastrawi stopword list.
3. **Kenapa stemming penting?** — Mengurangi sparsity fitur; `mengirim/dikirim/pengiriman` jadi 1 fitur.
4. **Kenapa TF-IDF bukan count saja?** — Count bias ke kata umum; IDF memberi bobot ke kata pembeda sentimen.
5. **Urutan boleh dibalik?** — Tidak. Stemming sebelum cleaning bisa gagal karena tanda baca menempel.

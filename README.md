# **Laporan Proyek _Machine Learning_ - Azri Andrizan**

---

# **_Predictive Analysis Machine Failure_**

# **Domain Proyek**

Industri manufaktur modern sering kali menghadapi tantangan besar dalam menjaga keandalan sistem operasional mesin mereka dan masalah paling ditakutkan adalah kegagalan mesin. Kegagalan mesin (_mechanical failure_) tentunya selain dari menyebabkan terhenti nya waktu produksi, juga dapat menyebabkan kerugian besar pada perusahaan dan bahkan akan menjadi resiko keselamatan bagi operator atau pekerja. Kegagalan mesin ini merupakan masalah yang sangat krusial dalam industri manufaktur, di Indonesia salah satu perusahaan yang memproduksi kain tenun pernah mengalami kegagalan mesin yang berdampak pada kecacatan produk, dalam artikel ilmiah [[1]] yang meneliti permasalahan ini menyatakan bahwa kegagalan mesin tersebut menyebabkan kualitas produk menurun dengan persentase kecacatan produk yang dihasilkan diatas 2%, tentunya ini akan menyebabkan kerugian yang besar bagi perusahaan jika tidak ditangani dengan tepat. Dalam penelitian ini juga berhasil mengungkapkan jenis kegagalan potensial yang terjadi berkaitan dengan kurangnya pemeliharaan atau perawatan mesin seperti keausan komponen mesin, gangguan sistem mekanik seperti sabuk penggerak yang longgar atau rusak juga turut menjadi penyebab kegagalan mesin dalam memproduksi kain tenun yang berkualitas.

Selain berdampak pada produk itu sendiri, kegagalan mesin juga bisa berdampak pada keselamatan para operator atau pekerja. Dikutip dari artikel oleh Indonesia Safety Center [[2]], tentang kecelakaan pabrik kimia yang pernah terjadi di salah satu pabrik di India pada tahun 2022. Beberapa diantara pekerja kehilangan nyawa akibat menghirup gas beracun dari kebocoran pada sistem pengaman. Salah satu penyebab dari kebocoran ini adalah kerusakan atau kegagalan peralatan, seperti pipa saluran yang mengalami korosi atau kegagalan dalam pengoperasian peralatan.

Untuk mengurangi risiko kegagalan mesin, terdapat beberapa pendekatan yang dapat diterapkan :
*   Pemeliharaan Preventif
    Pemeliharaan ini melibatkan inspeksi dan perawatan berkala pada peralatan, bahkan ketika belum ada tanda-tanda kerusakan. Tujuannya adalah untuk mencegah kerusakan sebelum terjadi [[3]].

*   Teknologi Monitoring _Real-Time_
    Dengan memanfaatkan sensor canggih dan sistem monitoring _real-time_, perusahaan dapat mengidentifikasi potensi masalah sebelum _downtime_ terjadi. Teknologi ini memungkinkan peringatan dini untuk peralatan yang mulai menunjukkan tanda-tanda kegagalan [[4]].

*   Pemeliharaan Prediktif (_Predictive Maintenance_)
    Pendekatan ini menggunakan algoritma _machine learning_ pada data historis yang dikumpulkan secara kontinu untuk memprediksi kapan kegagalan mungkin terjadi. Dengan demikian, perusahaan dapat merencanakan perbaikan atau penggantian komponen sebelum kegagalan terjadi, sehingga mengurangi waktu henti dan meningkatkan efisiensi operasional [[5], [6]].

# **_Business Understanding_**
## **_Problem Statement_**
Berangkat dari permasalahan yang telah dipaparkan pada bagian **Domain Proyek**, proyek ini berfokus pada identifikasi dan mengurangi risiko kegagalan mesin menggunakan pendekatan berbasis data dan algoritma machine learning. Adapun masalah spesifik yang akan dijawab adalah:
1.  Bagaimana memanfaatkan data historis mesin untuk mendeteksi potensi kegagalan mesin sejak dini?
2.  Bagaimana menangani ketidakseimbangan data sehingga kegagalan mesin yang jarang terjadi dapat tetap terdeteksi?
3.  Bagaimana hasil prediksi dapat digunakan untuk mendukung pengambilan keputusan terkait pemeliharaan preventif dan predictive maintenance?

## **_Goals_**
Proyek ini bertujuan untuk membangun model prediktif berbasis machine learning yang mampu mendeteksi potensi kegagalan mesin dengan akurasi tinggi. Tujuan utama ini dipecah menjadi beberapa sasaran spesifik:
1.  **Mengembangkan solusi berbasis _anomaly detection_** untuk mengidentifikasi penyimpangan operasional mesin yang dapat menjadi indikasi awal kegagalan.
2.  **Membangun model multiklasifikasi** untuk mengidentifikasi jenis kegagalan mesin secara spesifik berdasarkan parameter teknis dalam dataset.
3.  **Mengoptimalkan performa deteksi pada kelas minoritas**, yaitu jenis kegagalan mesin yang jarang terjadi, melalui penerapan teknik oversampling seperti SMOTE.
4. **Membandingkan performa dari 3 jenis model algoritma _machine learning_**, lalu memilih satu model terbaik berdasarkan metrik evaluasi yang digunakan

## **_Solution Statement_**
Untuk mencapai tujuan proyek, berikut adalah solusi yang diusulkan:
1.  **Dua Tahap Pemodelan**
    *   **Tahap 1: _Anomaly Detection_**
    _Anomaly detection_ diterapkan untuk mendeteksi pola operasional mesin yang menyimpang dari kondisi normal. Dengan pendekatan ini, potensi kegagalan dapat diidentifikasi lebih awal sebelum terjadi kerusakan signifikan.

    *   **Tahap 2: Multiklasifikasi**
    Pada tahap kedua, model multiklasifikasi dikembangkan untuk mengklasifikasikan jenis kegagalan mesin secara spesifik. Tiga algoritma _machine learning_ digunakan, yaitu **Random Forest**, **SVM**, dan **XGBoost**, untuk memastikan hasil prediksi yang akurat.

2.  **Penerapan Teknik _Oversampling_**

    Ketidakseimbangan data kegagalan diatasi dengan menggunakan **SMOTE** (_Synthetic Minority Oversampling Technique_). Teknik ini menciptakan data sintetis untuk kelas kegagalan minoritas, sehingga model lebih mampu mengenali pola kegagalan yang jarang terjadi.

3.  **Evaluasi Model dengan Metrik Terukur**
    
    Model dievaluasi menggunakan metrik berikut untuk memastikan efektivitas dan aplikabilitas:
    *   **AUC-ROC** (_Area Under the Receiver Operating Characteristic Curve_)untuk membedakan antara kelas positif dan negatif di berbagai _threshold_.
    *   **Precision** untuk menilai akurasi prediksi kegagalan.
    *   **Recall** untuk mengukur kemampuan model dalam mendeteksi kegagalan yang sebenarnya terjadi.
    *   **F1-score** sebagai metrik keseimbangan antara precision dan recall.
    *   **Confusion matrix** untuk memberikan insight mengenai distribusi kesalahan prediksi model.

# **_Data Understanding_**
Pada proyek ini, data historis kegagalan mesin yang digunakan diambil dari dataset yang berjudul **AI4I2020** yang disediakan oleh **UCI Machine Learning Repository**, dataset dapat diakses dan diunduh pada tautan [AI4I2020 Dataset](https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip). Dataset ini terdiri dari data simulasi yang dirancang untuk mereplikasi skenario dunia nyata terkait performa dan kegagalan mesin industri. Dataset ini menyediakan informasi berupa parameter operasional, data telemetri, dan kejadian kegagalan mesin yang dapat digunakan untuk mengidentifikasi pola yang mengarah pada kegagalan.

Dataset ini mendukung berbagai tugas analisis, seperti deteksi anomali, klasifikasi biner, dan klasifikasi multikelas. Dengan menggunakan dataset ini, pengguna dapat mengevaluasi pendekatan untuk memprediksi kapan kegagalan mesin mungkin terjadi atau menganalisis perilaku operasional mesin dalam kondisi normal dan abnormal.

Dataset disediakan dalam format csv (_comma seperated value_), berisi jumlah sampel dengan 10.000 baris data yang mencakup berbagai kondisi operasi dan status kegagalan mesin.

Terdapat 8 fitur utama dalam dataset ini, yang meliputi :
| Fitur         | Deskripsi                                          |Tipe Data|
|:--------------|:--------------------------------------------------|:---------|
|UDI| Unique Data Identifier. Nomor identifikasi untuk setiap entri data|int64|
|Product ID     |ID unik untuk setiap mesin atau produk              |object|
|Type           |Jenis mesin atau kategori produk (L, M, atau H)     |object|
|Air Temperature (K)|Temperatur udara dalam skala Kelvin |float64|
|Process Temperature (K)|Temperatur proses dalam skala Kelvin|float64|
|Rotational Speed (rpm)|Kecepatan rotasi dalam satuan RPM (_Rad per Minute_)|int64|
|Torque (Nm)|Torsi yang dihasilkan mesin dalam Newton-meter|float64|
|Tool Wear (min)|Waktu pemakaian alat dalam menit|int64|



Selain itu, terdapat 6 kolom target dalam dataset ini, yang meliputi :
|Target|Deskripsi|Tipe Data|
|:-----|:--------|:--------|
|Machine failure|Berisi nilai biner yang menunjukkan status kegagalan mesin. Nilai 0 menunjukkan mesin dalam kondisi normal, nilai 1 menunjukkan mesin mengalami kegagalan|int64|
|TWF|_Tool Wear Failure_. Kegagalan yang disebabkan oleh keausan alat atau komponen mesin yang digunakan dalam proses produksi|int64|
|HDF|_Heat Dissipation Failure_. Kegagalan ini terjadi ketika sistem mesin gagal dalam membuang panas dengan efektif. Akibatnya, suhu mesin menjadi terlalu tinggi, yang bisa menyebabkan kerusakan pada komponen internal mesin|int64|
|PWF|_Power Failure_.  kegagalan pada sistem pasokan daya|int64|
|OSF|_Overstrain Failure_. Kegagalan jenis ini terjadi ketika mesin mengalami tekanan atau beban yang lebih tinggi dari batas kapasitas yang telah ditentukan|int64|
|RNF|_Random Failure_. kegagalan yang tidak dapat diprediksi dan sering kali disebabkan oleh faktor-faktor acak atau kondisi lingkungan yang tidak terduga|int64|

## **__Exploratory Data Analysis__**

*   **_Missing Value Identification_**

    ![Gambar 1. Distribusi missing value pada kolom Tool Wear [min] ](https://github.com/azri-andrizan/Assets/blob/main/image_predictive_analysis/missing_value_distribution.PNG)
    Gambar 1. Distribusi _missing value_ pada kolom Tool Wear

    Pada variabel _Tool wear_ (min), ditemukan 120 baris data dengan nilai 0. Setelah dilakukan analisis lebih lanjut, nilai 0 pada 120 baris data tersebut dianggap tidak akurat atau tidak valid berdasarkan alasan berikut:
    1. Ketidaksesuaian dengan Kondisi Operasional Mesin. Secara logis, jika mesin menunjukkan _Process temperature_ dan _Rotational speed_ yang signifikan, artinya mesin sedang beroperasi. Namun, nilai _Tool wear_ (min) yang sama dengan 0 menit menunjukkan seolah-olah alat tidak digunakan. Hal ini menciptakan inkonsistensi karena alat seharusnya mengalami keausan (_tool wear_) selama mesin beroperasi.
    2. Tidak Ada Korelasi dengan Variabel Lain. Analisis lebih lanjut menunjukkan bahwa nilai _Tool wear_ (min) sebesar 0 tidak memiliki korelasi yang logis atau linier dengan variabel lain, seperti _Rotational speed_ atau _Process temperature_. Jika nilai tersebut valid, seharusnya ada pola hubungan yang dapat diidentifikasi. Ketiadaan korelasi ini semakin menguatkan asumsi bahwa nilai 0 merupakan anomali.
    3. Indikasi Penggunaan Mesin oleh Variabel Lain. Variabel lain, seperti _Process temperature_ dan _Rotational speed_, menunjukkan adanya aktivitas penggunaan mesin pada data yang memiliki nilai _Tool wear_ (min) sebesar 0. Ini menunjukkan bahwa data tersebut kemungkinan besar tidak mencerminkan kondisi sebenarnya.

    Sehingga 120 baris data yang dianggap _missing value_ ini dihapus dari _dataframe_ karena jumlah _missing value_ hanya sekitar 1,2% dari total data (10.000 sampel). Penghapusan ini dianggap tidak akan mempengaruhi analisis selanjutnya secara signifikan.

* **_Univariate Analysis_**

    * _Categorical Features_
    ![Gambar 2. Univariate Analysis Categorical Features ](https://github.com/azri-andrizan/Assets/blob/main/image_predictive_analysis/univariate_categorical.png)
    Gambar 2. _Univariate Analysis Categorical Features_

        Berdasarkan Gambar 2 diatas, tipe mesin dengan distribusi terbanyak adalah tipe L. Dataset membagi tipe mesin menjadi tiga kategori:
    
        - Tipe L (_Light_): Mesin ringan untuk beban rendah dan durasi operasi singkat, cocok untuk produksi skala kecil.
        - Tipe M (_Medium_): Mesin menengah untuk aplikasi dengan beban sedang dan operasi lebih lama, umum digunakan pada skala produksi menengah.
        - Tipe H (_Heavy_): Mesin berat untuk aplikasi intensif seperti produksi skala besar atau industri berat dengan durasi operasi tinggi.

        _Failure Type_ yang paling banyak distribusinya pada dataset adalah TWF (_Tool Wear Failure_) dengan distribusi sebesar 97 %. Berikut keterangan dari masing-masing Failure Type:
        - TWF (_Tool Wear Failure_) adalah kegagalan alat karena keausan.
        - HDF (_Heat Dissipation Failure_) adalah kegagalan karena masalah disipasi panas.
        - PWF (_Power Failure_) adalah kegagalan yang disebabkan oleh masalah daya.
        - OSF (_Overstrain Failure_) adalah Kegagalan yang disebabkan oleh tekanan berlebih.
        - RNF (_Random Failures_) adalah Kegagalan acak yang tidak terklasifikasikan ke dalam kategori kegagalan lainnya.

    * _Numerical Features_
    ![Gambar 3. Univariate Analysis Numerical Features ](https://github.com/azri-andrizan/Assets/blob/main/image_predictive_analysis/univariate_numerical.png)
    Gambar 3. _Univariate Analysis Numerical Features_

        Berdasarkan Gambar 3 diatas, dapat diambil beberapa informasi:
    
        - _Air Temperature_. Distribusi _air temperature_ mengindikasikan adanya beberapa kelompok dalam data. Tidak ada pola distribusi yang simetris dan terlihat sedikit tersebar. Nilai rentang suhu udara berkisar antara 294 K hingga 305 K, dengan sebagian besar data terkonsentrasi di sekitar 298–301 K.
        - _Process Temperature_. Distribusi suhu proses juga memiliki pola distribusi yang tersebar dengan puncak pada sekitar 310 K. Tidak terlihat adanya _outlier_ yang signifikan, dan distribusi ini menunjukkan bahwa sebagian besar mesin beroperasi pada suhu proses yang berada dalam rentang 308–312 K.
        - _Rotational Speed_. Distribusi kecepatan rotasi sangat condong ke kanan (_positively skewed_), dengan sebagian besar mesin beroperasi di sekitar 1500 rpm. Ini menunjukkan bahwa ada lebih banyak mesin dengan kecepatan rotasi yang lebih rendah dibandingkan kecepatan rotasi yang lebih tinggi.
        - _Torque_. Distribusi _torque_ berbentuk normal atau mendekati simetris, dengan puncak sekitar 40 Nm. Hal ini menunjukkan bahwa torsi mesin dalam dataset ini relatif terpusat di sekitar nilai tengah dan sebagian besar mesin bekerja pada kisaran torsi yang seragam.
        - _Tool Wear_. Distribusi _tool wear_ agak merata hingga mendekati 200 menit, kemudian turun tajam. Ini menunjukkan bahwa waktu pemakaian alat berkisar cukup luas, dan terdapat banyak mesin yang beroperasi dengan waktu pemakaian alat yang lebih rendah hingga menengah. Tidak ada indikasi _outlier_ yang signifikan, meskipun distribusi menurun drastis pada batas waktu pemakaian yang lebih tinggi.
        - _Machine Failure_. Distribusi biner (0 dan 1), di mana nilai 0 sangat mendominasi. Ini berarti kegagalan mesin jarang terjadi dalam dataset, dan sebagian besar mesin dalam kondisi baik (tidak mengalami kegagalan). Perbandingan jumlah antara kegagalan mesin (1) dan tidak ada kegagalan mesin (0) sangat tidak seimbang.

* **_Multivariate Analysis_**

    * _Correlation between Numerical Variable_
    ![Gambar 4. Multivariate Analysis Corelation Numerical Features ](https://github.com/azri-andrizan/Assets/blob/main/image_predictive_analysis/korelasi_numerical.png)
    Gambar 4. Korelasi antar Numerikal Variabel

        Pairplot pada Gambar 4 mengungkap pola hubungan antar fitur numerik dalam dataset, dengan wawasan berikut:
    
        - _Air Temperature vs. Process Temperature_: Hubungan linier positif menunjukkan bahwa peningkatan _Air Temperature_ sejalan dengan kenaikan _Process Temperature_, kemungkinan karena pengaruh kondisi operasional mesin.
        - _Process Temperature vs. Torque_: Hubungan _non-linier_ menurun menunjukkan bahwa pada suhu proses tinggi, _Torque_ cenderung rendah. Pola ini mungkin mencerminkan pembatasan torsi pada suhu tertentu.
        - _Rotational Speed vs. Torque_: Hubungan _non-linier_ menunjukkan bahwa peningkatan _Torque_ cenderung menurunkan _Rotational Speed_, kemungkinan sebagai mekanisme perlindungan mesin.
        - _Tool Wear_: Distribusi data yang menyebar tidak menunjukkan korelasi signifikan dengan fitur lain, namun nilai _Tool Wear_ tinggi dapat memengaruhi prediksi kegagalan mesin.
        - _Machine Failure_: Sebagian besar data menunjukkan mesin tidak gagal (_Machine Failure_ = 0). Kegagalan mesin relatif jarang, dan korelasi langsung dengan fitur lain tidak terlihat.
        - _Outliers_: Beberapa _outlier_ teridentifikasi pada _Process Temperature_ dan _Torque_, yang dapat menunjukkan kondisi ekstrem dengan potensi risiko kegagalan mesin.
    
    * _Bivariate Analysis of Categorical and Numerical Features_
    ![Gambar 5. Bivariate Analysis of Categorical and Numerical Features ](https://github.com/azri-andrizan/Assets/blob/main/image_predictive_analysis/bivariate_numerical_and_categorical.png)
    Gambar 5. Analisis _Bivariate_ antara _Numerical_ dan _Categorical Features_

        Dari _boxplot_ pada Gambar 5 diatas dapat disimpulkan :
        * _Numerical Features_ berdasarkan Tipe Mesin

            Berdasarkan _boxplot_ (Gambar 5), tidak ditemukan perbedaan signifikan antara tipe mesin (L, M, dan H) pada fitur _Air Temperature_, _Process Temperature_, dan _Tool Wear_. Hal ini menunjukkan bahwa tipe mesin tidak memengaruhi parameter tersebut secara signifikan. Namun, pada _Rotational Speed_ dan _Torque_, terdapat beberapa _outlier_, khususnya pada _Rotational Speed_. _Outlier_ ini dapat mengindikasikan potensi kerusakan atau variasi dalam performa operasional.

        * _Numerical Features_ berdasarkan jenis kegagalan

            Fitur _Rotational Speed_, _Torque_, dan _Tool Wear_ menunjukkan kemampuan yang lebih signifikan dalam memisahkan jenis kegagalan dibandingkan dengan _Air Temperature_ dan _Process Temperature_.
            - _Rotational Speed_ menjadi indikator utama untuk kegagalan PWF (_Power Failure_).
            - _Torque_ juga berkorelasi erat dengan kegagalan PWF.
            - _Tool Wear_ lebih relevan dalam mengidentifikasi kegagalan TWF (_Tool Wear Failure_), OSF (_Overstrain Failure_), dan HDF (_Heat Dissipation Failure_).
    
# **_Data Preparation_**
Pada tahap _data preparation_, beberapa langkah dilakukan untuk memastikan kualitas data, memfasilitasi analisis yang lebih baik, dan meningkatkan kinerja model. Tahapan yang dilakukan meliputi:

## **_One-Hot Encoding_** 
Fitur kategorikal seperti _Type_ (tipe mesin) dan _Failure Type_ (jenis kegagalan) diubah menjadi fitur numerik menggunakan _one-hot encoding_. Proses ini menghasilkan kolom baru untuk setiap kategori, dengan nilai biner yang menunjukkan keberadaan kategori dalam setiap observasi. Hal ini memudahkan model dalam memahami data kategorikal tanpa memberikan urutan atau bobot yang tidak diinginkan.

```python
# Mengubah fitur kategorikal kedalam bentuk one hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=['Type', 'Failure Type'])
df_encoded.head()
```

Dengan kode diatas maka masing-masing value pada kolom _Type_ dan _Failure Type_akan menjadi kolom baru, dan memiliki tipe data _boolean_ yang hanya terdiri dari nilai _True_ dan _False. Contoh hasil _one hot encoding_ dari kolom kategorikal tersebut seperti yang ditunjukkan tabel dibawah ini.

| Type_H| Type_L| Type_M| Failure Type_HDF| Failure Type_No Failure| ...|
|:------|:------|:------| :---------------| :----------------------| :--|
| False | True  | False | True            | False                  | ...|
| True  | False | False | False           | True                   | ...|
| False | False | True  | False           | True                   | ...|
| ...   | ...   | ...   | ...             | ...                    | ...|

Namun, value dari masing-masing kolom hasil _one hot encoding_ masih belum berbentuk numerik. Maka nilai _boolean_ ini diubah dari _True_ dan  _False_ menjadi 1 dan 0. 

```python
# Mengubah value boolean menjadi integer pada kolom hasil one hot encoding
boelan_columns = df_for_anomaly.select_dtypes(include=['bool']).columns
df_for_anomaly[boelan_columns] = df_for_anomaly[boelan_columns].astype(int)
```

Maka, hasilnya akan tampak seperti berikut.
| Type_H| Type_L| Type_M| Failure Type_HDF| Failure Type_No Failure| ...|
|:------|:------|:------| :---------------| :----------------------| :--|
| 0     | 1     | 0     | 1               | 0                      | ...|
| 1     | 0     | 0     | 0               | 1                      | ...|
| 0     | 0     | 1     | 0               | 1                      | ...|
| ...   | ...   | ...   | ...             | ...                    | ...|


## **Standarisasi Fitur Numerik**
Fitur numerik seperti suhu, tekanan, kecepatan rotasi, dan kelembaban distandarisasi sehingga memiliki _mean_ 0 dan _standard deviation_ 1. Langkah ini penting untuk menghindari bias yang mungkin timbul dari skala data yang berbeda.

```python
# Standarisasi value fitur numerik
from sklearn.preprocessing import StandardScaler

numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()

df_for_anomaly[numerical_features] = scaler.fit_transform(df_for_anomaly[numerical_features])
```
Berdasarkan _python_ diatas, nilai-nilai numerik pada kolom numerikal akan distandarisasi menggunakan _StandardScaler_. Sehingga value pada kolom numerikal yang sebelumnya seperti :
| Air temperature [K]| Process temperature [K]| ...|
|:-------------------|:-----------------------| :--|
| 298.2              | 308.7                  | ...|
| 298.1              | 308.5                  | ...|
| ...                | ...                    | ...|

menjadi seperti berikut :
| Air temperature [K]| Process temperature [K]| ...|
|:-------------------|:-----------------------| :--|
| -0.902753          | -0.880452              | ...|
| -0.95275           | -1.01526               | ...|
| ...                | ...                    | ...|

## **Pembuatan Target untuk Tahap Pertama (Deteksi Anomali)**
Karena distribusi data yang sangat _imbalanced_ dengan mayoritas label "_No Failure_," tahap pertama fokus pada klasifikasi biner untuk mendeteksi anomali. Proses ini membantu model membedakan kejadian kegagalan dari operasional normal.
```python
# Memisahkan fitur dan target untuk tahap 1 (anomaly detection)
x_anomaly = df_for_anomaly.drop(columns=['Machine failure', 'Failure Type_HDF', 'Failure Type_No Failure', 'Failure Type_OSF', 'Failure Type_PWF', 'Failure Type_TWF'])
y_anomaly = df_for_anomaly['Machine failure']
```

Dengan kode diatas, fitur dan target dipisahkan menjadi **x_anomaly** dan **y_anomaly**. Dimana **x** mewakili fitur dan **y** mewakili target. **x_anomaly** memiliki semua kolom yang ada pada dataframe terkecuali kolom _Machine failure_, _Failure Type_HDF_, _Failure Type_No Failure_, _Failure Type_OSF_, _Failure Type_PWF_, _Failure Type_TWF_. Sedangkan **y_anomaly** hanya memiliki kolom target yakni _Machine failure_.

## **Pemisahan Data untuk Pelatihan dan Pengujian**
Setelah _preprocessing_, data dibagi menjadi _training set_ dan _testing set_ dengan rasio tertentu, memastikan evaluasi model pada data yang tidak terlihat selama pelatihan untuk memberikan gambaran yang lebih akurat mengenai kinerjanya di dunia nyata.

```python
# Membagi data untuk training dan testing (binary classifaction)
from sklearn.model_selection import train_test_split

x_train_anomaly, x_test_anomaly, y_train_anomaly, y_test_anomaly = train_test_split(x_anomaly, y_anomaly, test_size=0.2, random_state=42, stratify=y_anomaly)
```

Kode diatas akan membagi dataset menjadi data untuk _training model_ dan _testing model_ untuk evaluasi model nantinya. Jadi **x_anomaly** di _split_ menjadi **x_train_anomaly** dan **x_test_anomaly**, begitu juga dengan **y_anomaly** di _split_ menjadi **y_train_anomaly** dan **y_test_anomaly**. Data _training_ dan _testing_ dibagi dengan rasio 80 : 20, dimana pada kode diatas ditunjukkan oleh parameter test_size pada train_test_split. test_size=0.2 berarti ukuran untuk data _testing_ adalah 20% sedangkan selebihnya adalah data _training_. Pembagian data ini bertujuan agar model dapat dilatih dan diuji dengan data yang berbeda, untuk memastikan saat pengujian model melihat data yang belum pernah dilihat sebelumnya pada proses _training_.

# **_Modelling_**
Pada bagian Modelling, proyek ini terbagi menjadi dua tahap pemodelan utama. Tahap pertama adalah Anomaly Detection, yang bertujuan untuk mendeteksi kejadian yang tidak biasa atau anomali dalam data. Pada tahap ini, model dikembangkan untuk memisahkan kejadian kegagalan dari operasional normal.

Tahap kedua adalah Multiclassification, di mana model dikembangkan untuk mengklasifikasikan jenis kegagalan mesin menjadi beberapa kategori berdasarkan fitur yang telah diproses. Setiap tahap pemodelan ini memainkan peran penting dalam memprediksi dan menganalisis kondisi mesin untuk meningkatkan efisiensi operasional.

## **_Modelling Anomaly Detection_**
Pada tahap **Anomaly Detection**, model _Random Forest_ digunakan untuk mendeteksi anomali dalam data. _Random Forest_ dipilih karena kemampuannya dalam menangani dataset yang besar dan tidak seimbang, serta kemampuannya dalam menangani variabel _input_ yang kompleks dan tidak linier. _Random Forest_ juga dikenal dengan stabilitas dan akurasi yang tinggi, membuatnya cocok untuk memisahkan data normal dari anomali.

Untuk evaluasi model, metrik yang digunakan meliputi _Classification Report_, _Confusion Matrix_, dan AUC-ROC. _Classification Report_ memberikan gambaran lengkap mengenai _precision_, _recall_, dan _f1-score_, yang sangat berguna dalam mengevaluasi performa model pada dataset yang sangat tidak seimbang. _Confusion Matrix_ digunakan untuk memberikan pemahaman yang lebih jelas tentang prediksi yang benar dan salah, sementara AUC-ROC mengukur kemampuan model dalam membedakan antara kelas positif dan negatif, yang penting untuk memahami seberapa baik model dapat mendeteksi anomali.

Model _Random Forest_ pada tahap ini menggunakan parameter **class_weight** sebagai solusi distribusi data yang _imbalanced_.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

rf_anomaly = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_anomaly.fit(x_train_anomaly, y_train_anomaly)
```

## **_Modelling Multiclass Classification_**
Pada tahap kedua, proses klasifikasi multikelas dilakukan untuk menentukan jenis kegagalan spesifik pada mesin yang sebelumnya telah terdeteksi sebagai anomali oleh model pada tahap pertama. Langkah ini bertujuan untuk mengidentifikasi kategori kegagalan, seperti HDF, PWF, OSF, atau TWF, berdasarkan fitur-fitur yang tersedia, sehingga dapat memberikan informasi yang lebih mendalam dan mendukung pengambilan keputusan lebih lanjut.

Pada tahap kedua, proses klasifikasi multikelas dilakukan untuk menentukan jenis kegagalan spesifik pada mesin yang sebelumnya telah terdeteksi sebagai anomali oleh model pada tahap pertama. Langkah ini bertujuan untuk mengidentifikasi kategori kegagalan, seperti HDF, PWF, OSF, atau TWF, berdasarkan fitur-fitur yang tersedia, sehingga dapat memberikan informasi yang lebih mendalam dan mendukung pengambilan keputusan lebih lanjut.

Proses ini mencakup beberapa langkah utama:

*   **Pemilihan fitur dan Penetapan Target**

    Dataset pada tahap ini hanya mencakup sampel yang telah diidentifikasi sebagai kegagalan (_anomaly_) pada tahap sebelumnya. Fitur-fitur yang memiliki relevansi tinggi dipilih kembali untuk memastikan model mampu memprediksi jenis kegagalan dengan akurasi optimal. Target klasifikasi adalah tipe kegagalan mesin dengan beberapa kelas yang merepresentasikan masing-masing jenis kegagalan.
    ```python
    # Menggunakan prediksi anomaly detection sebagai fitur baru
    df_for_anomaly['Anomaly Prediction'] = rf_anomaly.predict(df_for_anomaly.drop(columns=['Machine failure', 'Failure Type_HDF', 'Failure Type_No Failure', 'Failure Type_OSF', 'Failure Type_PWF', 'Failure Type_TWF']))
    ```
    
    Hasil dari model _anomaly detection_ dijadikan sebagai fitur baru pada dataframe untuk mendukung model _multiclass classification_ sehingga pada dataframe terdapat kolom baru yang menunjukkan _value anomaly prediction_. _Value_ tersebut bertipe _boolean_ dimana nilai 1 mewakili mesin yang dianggap _anomaly_ (mengalami kerusakan) dan nilai 0 mewakili mesin yang dianggap normal, seperti yang tampak pada kolom berikut.

    | ...| Failure Type_TWF| Anomaly Prediction|
    | :--| :---------------| :-----------------|
    | ...| 0               | 0                 |
    | ...| 1               | 1                 |
    | ...| ...             | ...               |

    Selanjutnya menentukan fitur dan target untuk model multiklasifikasi.
    ```python
    # Menentukan fitur dan target untuk model multiklasifikasi
    x_multiclass = df_for_anomaly.drop(columns=['Machine failure', 'Failure Type_HDF', 'Failure Type_No Failure', 'Failure Type_OSF', 'Failure Type_PWF', 'Failure Type_TWF'])
    y_multiclass = df_for_anomaly[['Failure Type_HDF', 'Failure Type_OSF', 'Failure Type_PWF', 'Failure Type_TWF']]
    ```

    Kode diatas membagi fitur dan target, dimana **x_multiclass** adalah fitur yang terdiri dari semua kolom terkecuali kolom *Machine failure*, *Failure Type_HDF*, *Failure Type_No Failure*, *Failure Type_OSF*, *Failure Type_PWF*, *Failure Type_TWF*. Sedangkan **y_multiclass** adalah target yang hanya terdiri dari kolom *Failure Type_HDF*, *Failure Type_OSF*, *Failure Type_PWF*, *Failure Type_TWF*.

*   **Pembagian Data _Training_ dan _Testing_**

    Data yang telah difilter berdasarkan jenis kegagalan dibagi menjadi dua subset, yaitu data training dan data testing. Pembagian ini dilakukan untuk memastikan bahwa model dapat dilatih secara efektif sekaligus dievaluasi dengan baik, guna menghindari overfitting.
    
    ```python
    # Split data training dan testing untuk model multiklasifikasi
    x_train_multiclass, x_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(x_multiclass, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass)
    ```

    Sama seperti pada tahap sebelumnya, data di _split_ menjadi data _training_ dan _testing_ dengan rasio 80 : 20, dimana 80 % adalah data _training_ dan 20 % adalah data _testing.


    Setelah dilakukan pembagian data menjadi data _training_ dan _testing_, selanjutnya dilakukan teknik _oversampling_. Hal ini dilakukan karena seperti yang diketahui adanya ketidakseimbangan distribusi data tiap kelasnya.

    ```python
    # Mengatasi imbalanced data dengan SMOTE
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    x_train_multiclass_resampled, y_train_multiclass_resampled = smote.fit_resample(x_train_multiclass, y_train_multiclass.values.argmax(axis=1))
    ```
*   **Pemilihan dan Pembangunan Model**

    Model klasifikasi multikelas yang digunakan mencakup **Random Forest**, **SVM**, dan **XGBoost**. Ketiga model ini dipilih karena memiliki performa yang andal dalam menangani data dengan distribusi kelas yang mungkin masih tidak seimbang (_imbalanced_). Kinerja masing-masing model akan dievaluasi berdasarkan metrik tertentu, seperti akurasi, F1-score, dan AUC-ROC, untuk menentukan model terbaik dalam mendeteksi jenis kegagalan mesin.

    ```python
    # Membangun model prediksi multiklasifikasi (jenis kegagalan pada mesin)
    # Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    rf_multiclass = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_multiclass.fit(x_train_multiclass_resampled, y_train_multiclass_resampled)
    y_pred_rf = rf_multiclass.predict(x_test_multiclass)
    
    # SVM Model
    from sklearn.svm import SVC
    svm_multiclass = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    svm_multiclass.fit(x_train_multiclass_resampled, y_train_multiclass_resampled)
    y_pred_svm = svm_multiclass.predict(x_test_multiclass)
    
    # XG Boost Model
    from sklearn.ensemble import GradientBoostingClassifier
    xgb_multiclass = GradientBoostingClassifier(random_state=42)
    xgb_multiclass.fit(x_train_multiclass_resampled, y_train_multiclass_resampled)
    ```

    Dari kode di atas, terlihat bahwa model **Random Forest** dan **SVM** menggunakan parameter *class_weight* untuk menangani data yang tidak seimbang (_imbalanced data_). Meskipun sebelumnya telah diterapkan teknik _oversampling_ untuk memperbaiki distribusi kelas, pembobotan tambahan melalui *class_weight* dilakukan sebagai langkah antisipasi terhadap potensi ketidakseimbangan yang mungkin masih terjadi setelah proses _oversampling_.

    Berbeda dengan kedua model tersebut, **XGBoost** tidak memerlukan parameter *class_weight*. Hal ini dikarenakan algoritma **XGBoost** secara inheren lebih tahan terhadap data yang tidak seimbang, berkat mekanisme pembobotan internalnya yang secara adaptif menyesuaikan bobot pada setiap iterasi untuk meminimalkan kesalahan prediksi pada kelas minoritas.

# **Model Evaluasi**
Untuk mengevaluasi performa model prediktif kegagalan mesin, digunakan dua metrik utama yang saling melengkapi, yaitu _classification report_ dan _confusion matrix_. Keduanya memberikan gambaran komprehensif tentang kinerja model dari berbagai aspek.

_Classification report_ digunakan untuk mengevaluasi performa model berdasarkan _precision_, _recall_, dan _F1-score_ pada setiap kelas, serta untuk menilai kinerja keseluruhan melalui nilai _macro average_ dan _weighted average_.

_Confusion matrix_ digunakan untuk menganalisis pola kesalahan prediksi, baik pada kelas dominan maupun minoritas.

Berikut adalah beberapa metrik yang digunakan pada proyek ini [[7], [8]]:
1. **Precision**
Precision mengukur sejauh mana prediksi positif model benar-benar positif. Dalam konteks multiklasifikasi, precision untuk setiap kelas dihitung sebagai:

$$
Precision = \frac{True Positives}{True Positives + False Positives}
$$

dimana:
    **True Positives** (TP) adalah jumlah sampel yang benar diklasifikasikan sebagai positif.
    **False Positives** (FP) adalah jumlah sampel yang salah diklasifikasikan sebagai positif.

2. **Recall**
Recall mengukur sejauh mana model dapat mendeteksi kelas positif yang sebenarnya. Recall untuk setiap kelas dihitung sebagai:

$$
Recall = \frac{True Positives}{True Positives + False Negatives}
$$

dimana :
    **False Negatives** (FN) adalah jumlah sampel yang salah diklasifikasikan sebagai negatif.

3. **F1-Score**
F1-Score adalah rata-rata harmonis dari precision dan recall, memberikan gambaran yang lebih baik tentang keseimbangan antara keduanya. Formula F1-Score adalah:

$$
F1 = 2 × \frac{Precision × Recall}{Precision + Recall}
$$

Metrik ini sangat berguna saat data tidak seimbang, karena mengimbangi precision dan recall.

4. **Accuracy**
Akurasi mengukur proporsi prediksi yang benar dari seluruh sampel:

$$
Accuracy = \frac{True Positives + True Negatives}{Total Samples}
$$

Namun, akurasi bisa menjadi metrik yang menyesatkan pada dataset yang tidak seimbang, karena model dapat memprediksi kelas mayoritas dengan akurasi tinggi meskipun gagal mendeteksi kelas minoritas.

5. **AUC-ROC**
Area Under the Curve - Receiver Operating Characteristic (AUC-ROC) adalah metrik yang menunjukkan kemampuan model untuk membedakan antara kelas positif dan negatif. AUC menggambarkan area di bawah kurva yang memplot True Positive Rate (TPR) versus False Positive Rate (FPR) untuk berbagai nilai threshold.
*   True Positive Rate (TPR) adalah sama dengan Recall.
*   False Positive Rate (FPR) dihitung sebagai:

$$
FPR = \frac{False Positives}{False Positives + True Negatives}
$$

Nilai AUC berkisar antara 0 dan 1. Nilai mendekati 1 menunjukkan bahwa model mampu memisahkan kelas dengan baik, sementara nilai 0.5 menunjukkan model acak tanpa kemampuan klasifikasi.

6. **Confusion Matrix**
Confusion matrix adalah tabel yang digunakan untuk menggambarkan kinerja model klasifikasi. Tabel ini menunjukkan jumlah prediksi yang benar dan salah pada setiap kelas dalam format matriks. Dalam kasus multiklasifikasi, matrix ini memperlihatkan:

*   True Positives (TP)
*   True Negatives (TN)
*   False Positives (FP)
*   False Negatives (FN)

Semua metrik yang disebutkan di atas dapat dihitung berdasarkan nilai-nilai dari confusion matrix.

## **Evaluasi Model _Anomaly Detection_**
Pada model _Anomaly Detection_ digunakan metrik _Classification report_ dan _Confusion Matrix_ untuk melihat evaluasi model.

_Classification Report_ :
|      | Precision| Recall| F1-Score|
| :----| :--------| :-----| :-------|
| 0    | 0.98     | 1.00  | 0.99    |
| 1    | 1.00     | 0.42  | 0.59    |
| Accuracy|       |       | 0.98    |
| Macro avg| 0.99 | 0.71  | 0.79    |
| Weighted avg| 0.98| 0.98| 0.98|

Dari tabel hasil _classification report_ diatas dapat diambil kesimpulan :
*   Kelas 0 (tidak gagal): _Precision_ 0.98 dan _recall_ 1.00, menunjukkan model sangat baik dalam mendeteksi mesin yang tidak gagal.
*   Kelas 1 (gagal): _Precision_ 1.00 tetapi _recall_ rendah (0.42), mengindikasikan model jarang salah memprediksi kegagalan namun sering gagal mendeteksi beberapa kasus kegagalan.
*   _Accuracy_: Akurasi keseluruhan mencapai 98%, tetapi lebih mencerminkan keberhasilan model dalam memprediksi kelas mayoritas karena distribusi data yang sangat tidak seimbang.

_Confusion Matrix_
|       | True| False|
| :-----| :---| :----|
| 0     | 1909| 0    |
| 1     | 39  | 28   |

Dari tabel hasil _confusion matrix_ diatas dapat diambil kesimpulan :
*   Kelas 0: Semua 1.909 sampel diklasifikasikan dengan benar.
*   Kelas 1: Dari 67 sampel, hanya 28 terdeteksi dengan benar sebagai kegagalan, sementara 39 salah diklasifikasikan sebagai tidak gagal.

Model menunjukkan performa baik pada kelas mayoritas, tetapi kesulitan dalam mendeteksi kegagalan (kelas minoritas) secara konsisten.

Karena recall rendah pada kelas 1, evaluasi dilanjutkan dengan metrik AUC-ROC untuk memberikan gambaran lebih seimbang. ROC memplot _True Positive Rate_ (TPR/recall) terhadap _False Positive Rate_ (FPR), sementara AUC mengukur kemampuan model membedakan antara dua kelas. Nilai AUC berkisar antara 0.5 (model acak) hingga 1 (model sempurna), dengan fokus pada meningkatkan deteksi kegagalan mesin.

![Gambar 6. Kurva ROC Anomaly Detection ](https://github.com/azri-andrizan/Assets/blob/main/image_predictive_analysis/roc_auc_anomaly.png)

Gambar 6. kurva ROC _Anomaly Detection_

Dari kurva pada Gambar 6 diatas dapat diambil kesimpulan :
*   AUC: Dengan nilai 0.97, model menunjukkan kemampuan tinggi dalam mendeteksi kegagalan (anomali) pada berbagai threshold, dengan tingkat false positives yang minimal. Hal ini menunjukkan efektivitas model dalam mencapai tujuan utama anomaly detection.
*   ROC Curve: Garis ROC yang mendekati sudut kiri atas mengindikasikan True Positive Rate (Recall) yang tinggi dengan False Positive Rate yang rendah, mencerminkan sensitivitas model yang baik terhadap kegagalan mesin tanpa mengurangi akurasi pada kelas tidak gagal.

## **Evaluasi Model Multiklasifikasi**
### **Model _Random Forest_**
_Classification Report_ :
|      | Precision| Recall| F1-Score|
| :----| :--------| :-----| :-------|
| 0    | 1.00     | 0.99  | 1.00    |
| 1    | 0.78     | 0.93  | 0.85    |
| 2    | 0.75     | 0.83  | 0.79    |
| 3    | 0.47     | 0.82  | 0.60    |
| Accuracy|       |       | 0.99    |
| Macro avg| 0.75 | 0.89  | 0.81    |
| Weighted avg| 0.99| 0.99| 0.99    |

Berdasarkan tabel _classification report_ diatas dapat diambil kesimpulan :
*   Model _Random Forest_ menunjukkan performa yang sangat baik secara keseluruhan, terutama pada kelas dominan (kelas 0). _Precision_, _recall_, dan _F1-score_ untuk kelas 0 mendekati sempurna dengan nilai masing-masing mencapai 1.00, 0.99, dan 1.00. Hal ini menunjukkan bahwa model ini sangat efektif dalam mengidentifikasi kelas dominan dengan tingkat kesalahan yang sangat rendah.

*   Untuk kelas 1, model ini menunjukkan kinerja yang cukup baik dengan _recall_ yang tinggi (0.93), meskipun _precision_-nya sedikit lebih rendah di angka 0.78. Ini menunjukkan bahwa ada beberapa kesalahan _false positive_ pada kelas ini, namun secara keseluruhan, model tetap berhasil mengidentifikasi sebagian besar sampel kelas 1 dengan benar.

*   Pada kelas 2, performa model masih cukup baik, dengan _precision_, _recall_, dan _F1-score_ masing-masing sebesar 0.75, 0.83, dan 0.79. Namun, untuk kelas 3, performa model menurun signifikan, dengan _precision_ yang hanya mencapai 0.47 dan _F1-score_ 0.60, meskipun _recall_ cukup tinggi di angka 0.82. Hal ini menunjukkan bahwa model cenderung memprediksi lebih banyak sampel sebagai kelas 3, sehingga menyebabkan banyak _false positives_. Secara keseluruhan, Random Forest menunjukkan performa yang kuat dan distribusi kesalahan yang cukup merata di semua kelas, meskipun kelas 3 tetap menjadi tantangan utama.

_confusion matrix_ :
| True/Predict| 0 | 1 | 2 | 3 |
| :-----------| :-| :-| :-| :-| 
| 0           |1916|2 | 5 | 9 |
| 1           | 0 | 14|0  | 1 |
| 2           | 2 | 1 |15 | 0 |
| 3           | 1 | 1 | 0 | 9 |

Berdasarkan tabel _confusion matrix_ diatas dapat diambil kesimpulan :
*   Kelas 0 (Failure Type_HDF) memiliki kinerja yang sangat baik pada kelas 0 dengan akurasi tinggi, karena prediksi untuk kelas ini mencapai 1916 benar dari 1932 total _instance_ (2, 5, dan 9 salah prediksi ke kelas lain). Ini menunjukkan bahwa model memiliki kemampuan yang baik untuk mengenali pola dalam kelas dominan.
*   Kelas 1 (Failure Type_OSF) memiliki 14 prediksi benar dari total 15 _instance_ untuk kelas ini, dengan hanya 1 _instance_ salah ke kelas lain.
*   Kelas 2 (Failure Type_PWF), dari total 18 _instance_, model mengklasifikasikan 15 dengan benar, meskipun terdapat kesalahan ke kelas 0 dan 1.
*   Kelas 3 (Failure Type_TWF) menunjukkan performa yang sedikit kurang optimal, karena ada 2 _instance_ yang salah diklasifikasikan ke kelas 0 dan 1, dengan hanya 9 _instance_ diklasifikasikan benar.

_Random Forest_ menunjukkan performa yang stabil, terutama pada kelas dominan (kelas 0). Meski demikian, ada beberapa kesalahan dalam mengklasifikasikan kelas minoritas (kelas 2 dan 3), tetapi kesalahan ini relatif sedikit.

### **Model SVM**
_Classification Report_ :
|      | Precision| Recall| F1-Score|
| :----| :--------| :-----| :-------|
| 0    | 1.00     | 0.92  | 0.96    |
| 1    | 0.65     | 1.00  | 0.79    |
| 2    | 0.53     | 1.00  | 0.69    |
| 3    | 0.07     | 0.82  | 0.12    |
| Accuracy|       |       | 0.92    |
| Macro avg| 0.56 | 0.94  | 0.64    |
| Weighted avg| 0.99| 0.92| 0.95    |

Berdasarkan tabel _classification report_ diatas dapat diambil kesimpulan :
*   Model Support Vector Machine (SVM) menunjukkan karakteristik yang berbeda dibandingkan Random Forest. Pada kelas 0, model ini bekerja dengan sangat baik, dengan precision sempurna di angka 1.00, namun recall sedikit lebih rendah (0.92), menghasilkan F1-score sebesar 0.96. Artinya, SVM mampu mengidentifikasi kelas ini dengan baik, meskipun ada beberapa sampel kelas 0 yang tidak terdeteksi.
*   Untuk kelas 1, model menunjukkan recall yang sempurna (1.00), namun precision turun menjadi 0.65, menghasilkan F1-score sebesar 0.79. Ini menandakan bahwa model sering salah mengklasifikasikan sampel dari kelas lain sebagai kelas 1. Fenomena serupa terjadi pada kelas 2, di mana recall kembali mencapai 1.00, namun precision turun menjadi 0.53. Sementara itu, performa model pada kelas 3 sangat buruk, dengan precision hanya sebesar 0.07 dan F1-score yang sangat rendah di angka 0.12, meskipun recall tetap cukup tinggi di 0.82. Hal ini menunjukkan bahwa model SVM cenderung sangat fokus pada recall tetapi mengorbankan precision, terutama pada kelas minoritas seperti kelas 3.

_confusion matrix_ :
| True/Predict| 0 | 1 | 2 | 3 |
| :-----------| :-| :-| :-| :-| 
| 0           |1784|7 |16 |125|
| 1           | 0 | 15|0  | 0 |
| 2           | 0 | 0 |18 | 0 |
| 3           | 1 | 1 | 0 | 9 |

Berdasarkan tabel _confusion matrix_ diatas dapat diambil kesimpulan :
*   Kelas 0, memiliki performa yang baik dengan 1784 prediksi benar dari 1932 total instance. Namun, terdapat 7, 16, dan 125 instance yang salah klasifikasi ke kelas lainnya, terutama ke kelas 3, yang menunjukkan SVM kurang mampu membedakan antara kelas 0 dan kelas 3.
*   Kelas 1, terdapat 15 prediksi benar dari total 15 instance, menunjukkan kemampuan klasifikasi yang baik untuk kelas ini.
*   Kelas 2 diklasifikasikan 18 dari 18 instance dengan benar , menunjukkan keunggulan SVM dalam menangani kelas ini.
*   Kelas 3 sama seperti kelas lainnya, terdapat sedikit instance yang salah klasifikasi pada kelas 3, dengan 9 instance diklasifikasikan benar.

SVM bekerja baik secara keseluruhan, namun terdapat beberapa masalah dalam membedakan antara kelas yang mirip, terutama kelas 0 dan 3, dengan jumlah kesalahan cukup besar ke kelas 3. Meski begitu, SVM unggul dalam menangani kelas 2, di mana seluruh instance diklasifikasikan dengan benar.

### **Model _XG Boost_**
_Classification Report_ :
|      | Precision| Recall| F1-Score|
| :----| :--------| :-----| :-------|
| 0    | 1.00     | 0.97  | 0.98    |
| 1    | 0.61     | 0.93  | 0.74    |
| 2    | 0.85     | 0.94  | 0.89    |
| 3    | 0.16     | 0.82  | 0.26    |
| Accuracy|       |       | 0.97    |
| Macro avg| 0.65 | 0.92  | 0.72    |
| Weighted avg| 0.99| 0.97| 0.98    |

Berdasarkan tabel _classification report_ diatas dapat diambil kesimpulan :
*   Model XGBoost memperlihatkan performa yang hampir mendekati Random Forest pada kelas dominan (kelas 0), dengan precision yang sempurna (1.00) dan recall sebesar 0.97, menghasilkan F1-score 0.98. Pada kelas 1, XGBoost memiliki performa yang cukup baik, dengan recall sebesar 0.93 dan precision di angka 0.61, menghasilkan F1-score sebesar 0.74. Namun, precision yang rendah menunjukkan bahwa model ini sering salah mengklasifikasikan sampel dari kelas lain sebagai kelas 1.
*   Pada kelas 2, XGBoost bekerja dengan sangat baik, dengan precision 0.85, recall 0.94, dan F1-score 0.89. Namun, seperti halnya SVM, performa model menurun drastis pada kelas 3. Precision pada kelas ini hanya mencapai 0.16, sementara recall tetap cukup tinggi di angka 0.82, menghasilkan F1-score yang rendah sebesar 0.26. Hal ini menunjukkan bahwa model cenderung membuat banyak kesalahan false positive saat memprediksi kelas 3.

_confusion matrix_ :
| True/Predict| 0 | 1 | 2 | 3 |
| :-----------| :-| :-| :-| :-| 
| 0           |1873|8 |3  | 48|
| 1           | 0 | 14|0  | 1 |
| 2           | 1 | 0 |17 | 0 |
| 3           | 1 | 1 | 0 | 9 |

Berdasarkan tabel _confusion matrix_ diatas dapat diambil kesimpulan :
*   Kelas 0, memiliki 1873 prediksi benar dari 1932 total instance, dengan beberapa kesalahan terutama ke kelas 3 (48 instance salah klasifikasi ke kelas ini).
*   Kelas 1, terdapat 14 prediksi benar dari total 15 instance pada kelas 1, dengan hanya 1 instance salah klasifikasi, menunjukkan performa yang baik pada kelas ini.
*   Kelas 2, dari 18 instance, XGradient Boosting mengklasifikasikan 17 dengan benar, menunjukkan performa yang kuat pada kelas ini.
*   Kelas 3, XGradient Boosting juga menunjukkan performa yang baik pada kelas ini dengan 9 prediksi benar dari total instance, dengan hanya 1 instance salah klasifikasi.

XGradient Boosting menunjukkan performa yang cukup baik, dengan akurasi tinggi pada sebagian besar kelas. Namun, ada beberapa kesalahan dalam membedakan kelas 0 dan 3, serupa dengan SVM, dengan kesalahan terbesar pada kelas 0 yang salah diklasifikasikan ke kelas 3.

## **Perbandingan Evaluasi Antar Model**
Perbandingan _Classification Report_ dari ketiga model:
|                 | Precision| Recall| F1-Score|
| :---------------| :--------| :-----| :-------|
| 0 RF            | 1.00     | 0.99  | 1.00    |
| 0 SVM           | 1.00     | 0.92  | 0.96    |
| 0 XGB           | 1.00     | 0.97  | 0.98    |
| 1 RF            | 0.78     | 0.93  | 0.85    |
| 1 SVM           | 0.65     | 1.00  | 0.79    |
| 1 XGB           | 0.61     | 0.93  | 0.74    |
| 2 RF            | 0.75     | 0.83  | 0.79    |
| 2 SVM           | 0.53     | 1.00  | 0.69    |
| 2 XGB           | 0.85     | 0.94  | 0.89    |
| 3 RF            | 0.47     | 0.82  | 0.60    |
| 3 SVM           | 0.07     | 0.82  | 0.12    |
| 3 XGB           | 0.16     | 0.82  | 0.26    |
| Accuracy RF     |          |       | 0.99    |
| Accuracy SVM    |          |       | 0.92    |
| Accuracy XGB    |          |       | 0.97    |
| Macro avg RF    | 0.75     | 0.89  | 0.81    |
| Macro avg SVM   | 0.56     | 0.94  | 0.64    |
| Macro avg XGB   | 0.65     | 0.92  | 0.72    |
| Weighted avg RF | 0.99     | 0.99  | 0.99    |
| Weighted avg SVM| 0.99     | 0.92  | 0.95    |
| Weighted avg XGB| 0.99     | 0.97  | 0.98    |


Perbandingan _Confusion Matrix_ dari ketiga model :

![Gambar 7 Perbandingan Confusion Matrix Antar Model](https://github.com/azri-andrizan/Assets/blob/main/image_predictive_analysis/perbandingan_confusion.png)

Gambar 7. Perbandingan _Confusion Matrix_ Dari Ketiga Model

Dari tabel _Classification Report_ dan _Confusion Matrix_ diatas dapat diambil kesimpulan :
*   **Model _Random Forest_**
    - _Performance Overview_ : 
    Model Random Forest menunjukkan performa yang stabil dalam mendeteksi kegagalan mesin pada dataset yang tidak seimbang. Penggunaan parameter *class_weight* membantu menyeimbangkan distribusi kelas, sehingga model mampu memberikan precision yang baik pada kelas minoritas. Namun, recall pada kelas ini tetap rendah, yang menunjukkan bahwa beberapa kasus tidak terdeteksi.
    - _Strengths_:
        * Akurasi tinggi pada kelas mayoritas
        * _Precision_ pada kelas minoritas cukup baik berkat pembobotan yang menyesuaikan distribusi kelas.
    - _Weakness_:
        * _Recall_ rendah untuk kelas minoritas, yang menunjukkan bahwa model masih melewatkan beberapa _instance_ penting.

*   **Model SVM**
    - _Performance Overview_ :
    Model SVM menggunakan kernel non-linear dan *class_weight* untuk menangani ketidakseimbangan kelas. _Precision_ pada kelas minoritas sangat tinggi, tetapi _recall_ tetap rendah, serupa dengan **Random Forest**. Ini menunjukkan bahwa meskipun SVM jarang salah memprediksi jenis kegagalan, model ini kesulitan dalam menangkap semua kasus jenis kegagalan.
    - _Strengths_:
        * Kemampuan mendeteksi kelas mayoritas dengan baik.
        * _Precision_ tinggi pada kelas minoritas, menunjukkan sedikitnya _false positives_.
    - _Weakness_:
        * Rendahnya _recall_ menunjukkan bahwa model kurang mampu mengenali semua jenis kegagalan.
        * Memerlukan waktu komputasi yang lebih lama dibandingkan model lainnya.

*   **Model _XG Boost_**
    - _Performance Overview_:
    **XGBoost**, yang secara bawaan tahan terhadap data tidak seimbang, memberikan hasil evaluasi yang cukup baik. Model ini menunjukkan keseimbangan antara _precision_ dan _recall_ pada kelas minoritas, namun masih cukup banyak terdapat kesalahan prediksi pada kelas mayoritas maupun minoritas.
    - _Strengths_:
        * Keseimbangan _precision_ dan _recall_ pada kelas minoritas.
    - _Weakness_:
        * Walaupun performanya lebih baik pada kelas minoritas, **XGBoost** tetap tidak sempurna dalam menangkap semua kegagalan pada kelas minoritas.

# **Conclusion**

Berdasarkan evaluasi dari _confusion matrix_ dan _classification report_, model **Random Forest** dipilih sebagai yang terbaik untuk tugas prediktif kegagalan mesin.

Alasan pemilihan :
1.  Kinerja Kelas Dominan (Class 0): **Random Forest** menunjukkan _precision_, _recall_, dan _F1-score_ yang hampir sempurna, mencerminkan kemampuan model menangani data mayoritas secara andal.
2.  Stabilitas Deteksi Kelas Minoritas: Model ini lebih stabil dalam mendeteksi kelas kegagalan meskipun dengan frekuensi rendah, menjadikannya lebih efektif dibanding **SVM** dan **XGBoost** pada dataset tidak seimbang.
3.  Peningkatan dengan **SMOTE**: Teknik oversampling seperti **SMOTE** meningkatkan nilai _recall_ pada kelas minoritas, menunjukkan kemampuan model menangkap lebih banyak _instance_ kegagalan yang sebelumnya terabaikan.
4.  Kinerja Keseluruhan: **Random Forest** memiliki nilai rata-rata _precision_, _recall_, dan _F1-score_ (_macro average_) yang unggul dibandingkan model lain, memberikan prediksi konsisten di semua kelas.

Model **Random Forest** dipilih karena kombinasi keandalannya pada kelas dominan, kemampuan mendeteksi kelas minoritas, dan kinerja keseluruhan yang lebih baik. Model ini cukup efektif untuk mendukung pengambilan keputusan berbasis data dalam pengelolaan operasional mesin, dengan potensi pengembangan lebih lanjut untuk peningkatan performa.


# **Referensi**
[[1]] A. Muslimah, "Analisis Keselamatan dan Kesehatan Kerja dalam Proses Produksi," Jurnal Gema Teknologi, vol. 11, no. 3, pp. 1–6, 2013. [Online]. Available: https://ejournal.undip.ac.id/index.php/jgti/article/view/6855. [Accessed: Nov. 17, 2024].

[[2]] Indonesia Safety Center, "Kasus Kecelakaan Kerja di Pabrik Kimia: Analisis Kecelakaan Gas Beracun yang Menggemparkan," 2021. [Online]. Available: https://indonesiasafetycenter.org/kasus-kecelakaan-kerja-di-pabrik-kimia-analisis-kecelakaan-gas-beracun-yang-menggemparkan/. [Accessed: Nov. 17, 2024].

[[3]] I. B. Gertsbakh, Models of Preventive Maintenance. Amsterdam: North-Holland Publishing, 1977. [Online]. Available: https://scholar.google.com/scholar_lookup?title=Gertsbakh:+Models+of+Preventive+Maintenance&author=Gertsbakh,+I.B.&publication_year=1977. [Accessed: Nov. 17, 2024].

[[4]] M. Pech, J. Vrchota, and J. Bednář, "Predictive Maintenance and Intelligent Sensors in Smart Factory: Review," Sensors, vol. 21, no. 4, pp. 1470, Feb. 2021. [Online]. Available: https://doi.org/10.3390/s21041470. [Accessed: Nov. 17, 2024].

[[5]] T. Zonta, C. A. Da Costa, R. Da Rosa Righi, M. J. De Lima, E. S. Da Trindade, and G. P. Li, "Predictive maintenance in the Industry 4.0: A systematic literature review," Computers & Industrial Engineering, vol. 150, pp. 106889, Aug. 2020. [Online]. Available: https://doi.org/10.1016/J.CIE.2020.106889. [Accessed: Nov. 17, 2024].

[[6]] D. Natanael and H. Sutanto, "Machine Learning Application Using Cost-Effective Components for Predictive Maintenance in Industry: A Tube Filling Machine Case Study," Journal of Manufacturing and Materials Processing, vol. 6, no. 5, pp. 108, Sep. 2022. [Online]. Available: https://doi.org/10.3390/jmmp6050108. [Accessed: Nov. 17, 2024].

[[7]] "Introduction to Precision, Recall, F1 score, and ROC," Towards Data Science, [Online]. Available: https://towardsdatascience.com. [Accessed: Nov. 17, 2024].

[[8]] Jason Brownlee, "AUC-ROC Curve in Machine Learning," Machine Learning Mastery, [Online]. Available: https://machinelearningmastery.com. [Accessed: Nov. 17, 2024].





[1]: https://ejournal.undip.ac.id/index.php/jgti/article/view/6855
[2]: https://indonesiasafetycenter.org/kasus-kecelakaan-kerja-di-pabrik-kimia-analisis-kecelakaan-gas-beracun-yang-menggemparkan/
[3]: https://scholar.google.com/scholar_lookup?title=Gertsbakh:+Models+of+Preventive+Maintenance&author=Gertsbakh,+I.B.&publication_year=1977
[4]: https://scholar.google.com/scholar_lookup?title=Predictive+Maintenance+and+Intelligent+Sensors+in+Smart+Factory:+Review&author=Pech,+M.&author=Vrchota,+J.&author=Bedn%C3%A1%C5%99,+J.&publication_year=2021&journal=Sensors&volume=21&pages=1470&doi=10.3390/s21041470&pmid=33672479
[5]: https://scholar.google.com/scholar_lookup?title=Predictive+maintenance+in+the+Industry+4.0:+A+systematic+literature+review&author=Zonta,+T.&author=Da+Costa,+C.A.&author=da+Rosa+Righi,+R.&author=de+Lima,+M.J.&author=da+Trindade,+E.S.&author=Li,+G.P.&publication_year=2020&journal=Comput.+Ind.+Eng.&volume=150&pages=106889&doi=10.1016/J.CIE.2020.106889
[6]: https://scholar.google.com/scholar_lookup?title=Machine+Learning+Application+Using+Cost-Effective+Components+for+Predictive+Maintenance+in+Industry:+A+Tube+Filling+Machine+Case+Study&author=Natanael,+D.&author=Sutanto,+H.&publication_year=2022&journal=J.+Manuf.+Mater.+Process.&volume=6&pages=108&doi=10.3390/jmmp6050108
[7]: https://towardsdatascience.com/
[8]: https://machinelearningmastery.com/

#Created by Kresna F a.k.a Amur Tigro
#Creation Date : 7 August 2025
#Version : 1.1
#This feature generator was created to help data analyst and scientist discover hidden insight using statistical calculations and preprocessing
import numpy as np
import pandas as pd
import re
import math
import emoji
from scipy.stats import beta
from collections import Counter
from itertools import combinations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class FeatureGenerator:
    def __init__(self):
        pass

    def official_brand_recognizer(self, brand_list, store_name):
        store_name_lower = store_name.lower().strip()
        for brand in brand_list:
            brand_lower = brand.lower().strip()
            if brand_lower in store_name_lower:
                if 'official' in store_name_lower:
                    return "Official Store"
                else:
                # kalau tidak ada yang cocok, baru return "Reseller"
                    return "Reseller"
        return "Reseller"
    
    def DataDistributions(self,df,column_name:str,bin_method:str,column_target_name:str,bins:int = None) -> pd.DataFrame:
        """
        Buat kolom berisi rentang (bin) harga untuk analisis distribusi penjualan brand.
        
        Parameters
        ----------
        data : str | Path | pd.DataFrame
            - Path/URL ke file CSV **atau**
            - DataFrame yang sudah ada di memori.
        column_name : str
            Nama kolom baru yang akan menyimpan label rentang harga.
        column_target_name : str
            Nama kolom berisi harga di DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Salinan DataFrame dengan kolom rentang harga (`column_name`) yang baru. """
        price_series = df[column_target_name].dropna()
        #search for local maxima and minima
        local_max = price_series.max()
        local_min = price_series.min()

        #binning method
        if bin_method == 'sturges':
            k_bins = int(1+3.3* np.log(len(df[column_target_name])))
            bin_edges = np.linspace(local_min,local_max, k_bins + 1)

        elif bin_method == 'quantile':
            bin_edges = price_series.quantile(np.linspace(0,1, bins+1 if bins else 5 )).unique()
            bin_edges.sort()
        elif bin_method == 'fixed':
            if not bins:
                bins = 5
            bin_edges = np.linspace(local_min, local_max, bins + 1)
        else:
            raise ValueError("Unknown Binning Method")
        
        if len(bin_edges)<2 or np.any(np.diff(bin_edges)==0):
            df[column_name] = df[column_target_name]
            return df
        
            #cutting price into segmented bins
        df[column_name] = pd.cut(df[column_target_name], bins = bin_edges, include_lowest = True).apply(lambda iv: f"{iv.left:,.0f}-{iv.right:,.0f}")

        return df
    def outlier_detector_IQR(self,df, column):
        """
        Works by using Interquartile Range(IQR).
        Data that are located outside the function are considered as "outliers"
        ----------------------------------------------------------------------------
        Input:
        df:pd.DataFrame
        DataFrame that filled with numeric data
        column :str
        "Column : Name of Column that will be analyzed the outlier"
        -----------------------------------------------------------------------------
        Returns :
        upper outlier (INT), 
        lower outlier(INT), 
        lower_bound(TUPLE),
        upper_bound (TUPLE)
        """
        DataSeries = pd.Series(df[column])
        Q1 = DataSeries.quantile(0.25)
        Q3 = DataSeries.quantile(0.75)

        #define the interquartile equation
        IQR = Q3-Q1
        lower_bound = Q1-1.5*IQR
        upper_bound = Q3+1.5*IQR
        if lower_bound < 0:
            lower_bound = 0
        else:
            lower_bound
        lower_outlier = DataSeries[(DataSeries<lower_bound)]
        upper_outlier = DataSeries[(DataSeries>upper_bound)]

        return lower_outlier,upper_outlier,(lower_bound, upper_bound)
    
    def volume_scanner(self,text):
        """
        Help you discover volumetric of SkinCare Products
        """
        text = str(text)
        pattern = r"(\d+)\s*(ml|mL|ML|Ml)"
        match = re.search(pattern, text,re.IGNORECASE)
        if match:
            net_volume = int(match.group(1))
            return f'{net_volume}'
        else:
            return None
    def extract_attributes(self,category,product_name, attributes_dict):
        """
        Extract product atributes from string using predefined keywords in a dictionary

        product_name : type -> str (String data that contain product name and its attributes)
        attributes_dict:type -> Dict(Dictionary files that contains predefined keywords for categorizing and extracting attributes from product name)

        """
        name = product_name.lower()
        detected_attributes = []
        #take subdictionary based on category
        subcats = attributes_dict.get(category, {})

        for attribute, keywords in subcats.items():
            for kw in keywords:
                if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", name):
                    detected_attributes.append(attribute)
                    break
        return detected_attributes
        
    def weight_scanner(self,text):
        """
        Help you discover weight of SkinCare/Cosmetics Products
        """    
        text = str(text)
        pattern = r"(\d+)\s*(gr|g|GR|gram|Gram|lb|lbs)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            weight = int(match.group(1))
            return f'{weight}'
        else:
            return None
          
    def spf_scanner(self,text):
        """
        If you want to work with Cosmetic Data, especially Sunscreen, this function will help you discover the SPF value
        """
        pattern = r"spf[\s\-]?(\d{2})(\+?)"
        match = re.search(pattern,text,re.IGNORECASE)
        if match:
            spf_value = int(match.group(1))
            if 15<= spf_value <=100:
                return f"{spf_value}{match.group(2)}"
        return None
    def shorten_name(self,name, max_len=50):
        name = str(name)
        return name if len(name) <= max_len else name[:max_len]+ '...'
    
    def product_categorizer(self,category_dict:dict, df,PRODUCT_COLUMN:str):
        """

        Args
        1.category_dict : Berisi dictionary dari penggolongan/kategorisasi produk-produk, buat sendiri dan definsikan keywordsnya dalam
          bentuk 
            dict = {'key':[values:keywords],'key2 : [keywords2],'key3':[keywords3]} >> DEFINISIKAN KEYWORDS dalam STRING
        2.df = Isikan DataFrame berikut juga nama kolom dimana data hasil kategorisasi akan diisikan
        3.PRODUCT_COLUMN : str >> Berisi kolom dimana terdapat nama produk. Nama kategori produk akan disesuaikan dengan category dictionary.
        """

        df[PRODUCT_COLUMN] =  df[PRODUCT_COLUMN].astype(str).str.lower()

        categories = {v.lower():k for k, lst in category_dict.items() for v in lst}

        pattern = '|'.join(re.escape(v) for v in categories.keys())

        matched = df[PRODUCT_COLUMN].str.extract(f"({pattern})",flags = re.IGNORECASE)[0].str.lower()

        categorized = matched.map(categories).fillna("Unknown")
        
        return categorized
       
    def CountUnitProduct(self,text):
        """
        --Count Amount of Unit in a Product Based on Product Name--
        text: str >> Product Name that will be scanned for product amount
        return count, 
        count_plus, if it can't be scanned 
        resulting 1
        """
        pattern = r'(\d+)\s*(pcs|pieces|unit|sachet|capsules|kapsul|tablet|kaplet|pil|capsule|strips|strip)'
        text = str(text).lower()
        
        matches = re.findall(pattern,text)
        
        if matches:
            for match in matches:
                return {"jumlah":int(match[0]),"satuan":match[1]}
        elif '+' in text:
            count_data = text.count('+') + 1
            return {"jumlah":count_data,"satuan":"mix/package"}
        return {"jumlah":1,"satuan":"Unknown"}

    def OutlierAnalyzer(self, df,brand_column,sales_column,graph_title,product_column,top_n):
        """
        Menghasilkan titik-titik yang  menjadi outlier dalam bentuk list untuk brand analzyer

        Args:
        df :pd.Dataframe
        Dataframe yang sudah terisi nilai

        brand_column:str
        Kolom brand product

        sales_column:str
        Kolom sales dari DataFrame

        graph_title:str
        Nama box-plot

        product_column:str
        Kolom Nama Produk di Dataframe

        top_n:int
        Deklarasikan nilai N untuk mencari top - n values dari nilai sales

        Returns:
        Box Plot

        """
        top_5_brand = (df.groupby(brand_column)[sales_column].sum().sort_values(ascending = False).head(top_n).index)
        top_brand_df = df[df[brand_column].isin(top_5_brand)]

        #create subplots
        n_brands = len(top_5_brand)
        cols = 2
        rows = math.ceil(n_brands/cols)

        fig_outlier = make_subplots(rows = rows,
                            cols = cols,
                            subplot_titles = top_5_brand,
                            vertical_spacing = 0.15)
        row =col = 1
        all_annotations = []
        for brand in top_5_brand:
            brand_df_filtered = top_brand_df[top_brand_df[brand_column] == brand]
        
            fig_outlier.add_trace(
                go.Box(
                    y = brand_df_filtered[sales_column],
                    boxpoints = "outliers",
                    jitter = 0.5,
                    pointpos = 0,
                    text = brand_df_filtered[product_column],
                    hoverinfo = "text + y",
                    name = brand),
                    row = row,
                    col = col)
            col += 1
            if col > cols:
                col = 1
                row += 1
        #updateing layout
        fig_outlier.update_layout(
            height = rows * 400,
            width = 1200,
            title = graph_title,
            showlegend = False,
            plot_bgcolor = 'white'
        )
        fig_outlier.show()
    class AdsAnalyzer():
        def __init__(self):
            pass
        def parse_info(self,entry):
            """
            Change dictionary data with certain delimiter in 1 cell of Excel into structured and operable dictionary
            Apply to a column of DataFrame
            entry : -> df[column_name].apply(lambda text:parse_info(text)) <- How to access entry in DataFrame File
            delimiter :str -> Substring that limits the data in each entry
            """
            result = {}
            parts = entry.split(';')
            for p in parts:
                if":" in p:
                    key,val = p.split(":")
                    key = key.strip()
                    val = float(val.strip().replace(",",""))
                    result[key] = val
            return result
        
    #Bayesian Feature Generator Segments
    class BayesianFeatureGen():
        def __init__(self):
            pass

        def credible_interval_beta(self, success_count:int, total:int, a = 1.0, b = 1.0, ci = 0.95):
            """
            Menghasilkan rata-rata dan credible interval dengan menggunakan distribusi Beta-Binomial

            Args:
            success_count:int (Hitung jumlah kejadian sukses)

            total :int (hitung jumlah total kejadian)

            a = 1(default) (Tentukan prior untuk distribusi Beta, untuk non-informative prior tulis 1)

            b = 1 (default) (Tentukan prior untuk distribusi Beta untuk non-informative prior tulis 1)

            ci = 0.95 (default) (Tentukan credible interval untuk peluang Bayesian-mu, CI = 0.95 akan menghasilkan batas bawah dari 2.5% hingga 97.5%)
            
            Return
            mean : float (Nilai rata-rata posterior)

            lo : float (nilai batas bawah dari credible interval)

            hi : float (nilai batas atas dari credible interval)

            a_post = float (nilai posterior dari kejadian sukses)

            b_post = float (nilai posterior dari kejadian tidak sukses)

            """
            a_post = a + success_count

            b_post = b + (total - success_count)

            alpha = 1- ci

            lo = beta.ppf(alpha/2, a_post, b_post)

            hi = beta.ppf(1-alpha/2, a_post, b_post)

            mean = a_post / (a_post + b_post)
            
            return mean, lo, hi,a_post, b_post
    
        def calculate_binomial_ci(self, df:pd.DataFrame, col1:str, col2:str, success_label:str, ci:int, prior_a = 1.0, prior_b = 1.0 ):
            """
            Hitung credible interval dengan menggunakan kategori tertentu

            Args:
            df : pd.DataFrame (Dataframe data yang digunakan untuk menghitung CI),

            col1:str (Kondisi I berdasarkan Bayesian Theorem)

            col2:str (Kondisi II berdasarkan Bayesian Theorem)

            label:str (Hitung peluang terjadinya kejadian sukses, masukkan sebagai kondisi yang ingin dicari CI-nya)

            ci = Credible Interval (Rentang Terpercaya, dibentuk berdasarkan fungsi distribusi data tertentu)

            prior_a:1(default) (prior untuk menetapkan nilai alpha)

            prior_b:1(default) (prior untuk menetapkan nilai beta)

            Returns:
            result:pd.DataFrame

            """
            #Hitung k dan n per bucket
            grp = df.groupby(col1).apply(
                lambda g:pd.Series({
                    "n":len(g),
                    "k":(g[col2] == success_label).sum()
                })
            ).reset_index()

            rows = []
            for _,r in grp.iterrows():
                mean,lo,hi, a_post, b_post = self.credible_interval_beta(
                    success_count = int(r["k"]),
                    total = int(r["n"]),
                    a = prior_a, b = prior_b, ci = ci
                )
                rows.append({
                    col1 : r[col1],
                    "success_count":int(r['k']),
                    "total" : int(r["n"]),
                    "posterior_mean":round(mean*100,2),
                    f"ci_{ci*100}_low":round(lo*100,2),
                    f"ci_{ci*100}_hi":round(hi*100,2),
                    "a_post":a_post,
                    "b_post":b_post
                })

            result = pd.DataFrame(rows).sort_values(by = col1)
            return result
    class DataViz():
        def __init__(self):
            pass
        def plot_spider(self,data, category,columns=None,**kwargs):
            """
            category:type-str. Category must exist in data as an Index.
            data:type->DataFrame. Amount of data that will be inputted into the spider chart
            columns:type-List. List of columns that exist in data that will be used as axis in spider chart
            """
            if columns is None:
                columns = data.columns.tolist()

            values = data.loc[category,columns].values.tolist()
            values += values[:1]

            categories = columns + [columns[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r = values,
                theta = categories,
                fill = 'toself',
                fillcolor = 'rgba(255, 165, 0, 0.3)',
                line = dict(color = 'red', width = 2),
                name = category
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis = dict(visible = True, range =[0,1])
                ),showlegend = True,
                title = f'Spider Chart - {category.title()}',
                width = 900,
                height = 600
            )

            fig.show()
            
    class NaturalLanguageProcessing():
        def __init__(self):
            pass
        def ExtractProductIngredients(self,text):
            # !ONLY WORK IN SHOPEE COSMETIC PRODUCT! #
            text = text.lower()
            pattern = re.compile(
                r'(?:\[\s*bahan utama\s*\]|hero ingredients\:|hero ingredient\:|komposisi|ingredients|komposisi unggulan|powerful ingredients & benefit|dengan kandungan|bahan utama)' # header
                r'([A-Za-z0-9,.\-\s()%\/]+?)' # Tangkap Teks setelah header
                r'(?=(?:\[\s*PENGIRIMAN\s*\]|No\.|tanggal|berat|formulasi|jumlah|cara pemakaian|cara penggunaan|pengiriman|cara penggunaan serum))',
                flags = re.IGNORECASE | re.DOTALL
            )

            matches = pattern.findall(text)
            if not matches:
                return 'No Data'
            ingredients = []

            for m in matches:#Bersihkan dari spasi, newline, dan karakter
                cleaned = re.sub(r'\s+',' ',m).strip()

                parts = [i.strip() for i in cleaned.split(',') if i.strip()]
                ingredients.extend(parts)

            return ingredients
        
        def ReviewCategory(self,text,dictio):
            # Use the product review text into certain categories 
            found_cats = []
            text_lower = text.lower()
            for cat, keywords in dictio.items():
                if any(kw in text_lower for kw in keywords):
                    found_cats.append(cat)
            return found_cats
        
        def get_cooccurence(self,text,window):
            # To find the co-occurence of n pair word in a sentence and in a documents
            """ Input 
            1.Text:str
            Text input that will be counted the word co-occurence

            2.Window 
            Amount of word pairs that will be analyzed 
            """
            if not isinstance(text, str):
                return Counter()
            text = text.strip()

            #Jaga2 kalo ngga ada datanya.
            if text == '':
                return Counter()
            
            #Lakukan Tokenization
            tokens = text.lower().split()

            #Kembalikan Counter() kalau panjang token < 2
            if len(tokens) <=2:
                return Counter()
            
            cooccur = Counter()

            if window is None or window >= len(tokens):
                pairs = combinations(tokens,2)

            elif window == 2:
                pairs = []
                for i in range(len(tokens)):
                    for j in range(i+1, min(i+window+1, len(tokens))):
                        w1,w2 = ((tokens[i], tokens[j]))
                    pair = tuple(sorted([w1,w2])) 
                    cooccur[pair] += 1
                return cooccur
            else:
                pairs = []
                for i in range(len(tokens)):
                    for j in range(i+1, min(i+window, len(tokens))):
                        for k in range(j + 1, min(j+window, len(tokens))):
                            w1,w2,w3 = ((tokens[i], tokens[j], tokens[k]))

                    pair = tuple(sorted([w1,w2,w3])) 

                    cooccur[pair] += 1

                return cooccur
        def get_monogram(self,text):
            if not isinstance(text,str) or text.strip() == "":
                return Counter()
            tokens = text.lower().split()
            return Counter(tokens)
        
        
        
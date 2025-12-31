import pandas as pd
import re
import emoji
import string
import numpy as np
from rapidfuzz import process,fuzz

class DataCleaner:
    def __init__(self):
        pass
    
    def remove_non_numeric(self,text):
        """
        Menghapus semua karakter kecuali angka dan titik desimal
        """
        return ''.join(re.findall(r'[\d\.]+', str(text)))
    
    def substring_remover(self,text,substrings:list,substitute_with:str):
        """
        Menghapus karakter yang diinginkan dari sebuah teks
        """
        text = str(text)
        for substring in substrings:
            text = text.replace(substring,substitute_with)
        return text
    def productname_clean(self,text):
        """
        Menghapus semua substring dalam tanda kurung siku [ ... ] dari sebuah string.
        """
        text = str(text)
        # Hapus semua [....] termasuk tanda kurungnya
        cleaned = re.sub(r'\[.*?\]', '', text)
        # Hilangkan spasi berlebih setelah penghapusan
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def decimal_to_thousands(self,val):
        val = str(val)
        if '.' in val and len(str(val)) >= 5:
            val = val.replace('.','')
            val = int(val)
            return val
        elif '.' in val and len(val) <=4:
            val = int(float(val))
            return val
        else:
            return int(val)
                
    def IndonesianCurrencyFormatter(self,df,column, return_type = 'float'):
        """
        Fungsi untuk mengubah menjadi format Indonesia 1.234,567 ('.' = Ribuan, ',' = desimal)
        1.df:pd.DataFrame 
        -Masukkan nama Dataframe
        2.column:str
        -Masukkan nama kolom yang ingin diubah formatnya 
        3.return_type = str
        """
        series = df[column].astype(str)

        #Replace sries NaN values dengan np.nan
        series  = series.replace(['nan','None',''],np.nan)
        mask = series.notna()
        #Lakukan proses masking
        if mask.any():
            # Bersihkan karakter selain digit, titik, koma, dan minus
            series.loc[mask] = series.loc[mask].str.replace(r'[^\d.,-]', '', regex=True)

        def convert_indonesian(value):
            if pd.isna(value) or value == '':
                return np.nan
            try:
                is_negative = value.startswith('-')
                if is_negative:
                    value = value[1:]

                #jika ada koma, pisahkan desimal
                if ',' in value:
                    parts = value.rsplit(',',1)
                    integer_part = parts[0].replace('.','')
                    decimal_part = parts[1]
                    result = f"{integer_part}.{decimal_part}"
                else:
                    result = value.replace('.','')
                return float(result)
            except:
                return np.nan
            
        series.loc[mask] = series.loc[mask].apply(convert_indonesian)

        if return_type == 'int':
            series = series.round().astype('Int64')  # Nullable integer
        else:
            series = series.astype('float64')
        
        return series
    def SplitRange(text:str,separator:str):
        """
        Memisahkan data yang menyatu karena adanya separator
        """
        text = str(text)
        if separator in text:
            text_1 = text.split(separator)[0]
            text_2 = text.split(separator)[1]
            return int(text_1),int(text_2)
        else:
            return text,text
    
    class NLPCleaner():
        def __init__(self):
            pass

        def CleaningText(self,text):
            review = text.lower()
            #hapus emoji
            review = emoji.demojize(review)
            review = re.sub(':[A-Za-z_-]+:', ' ', review)
            #hapus emoticon
            review = re.sub(r"([xX;:]'?[dDpPvVoO3)(])", ' ', review)
            #hapus username
            review = re.sub(r'@[A-Za-z0-9]+','',review)
            #hapus hashtag
            review = re.sub(r'#[A-Za-z0-9]+','',review)
            #hapus link/URL
            review = re.sub(r"http\S+",'',review)
            #hapus karakter khusus
            review = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", review)
            #hapus angka
            review = re.sub(r'\d+'," ",review)
            #Hapus kata "nya" dan "di"
            review = re.sub(r'nya\b','',review)
            review = re.sub(r'^di(?=\w)', '',review)

            return review
        
        def synonym_normalization(self,text, JSON_DATA):
            "Normalize data that contained synonyms"
            words = text.lower().split()
            normalized = []
            for w in words:
                normalized.append(JSON_DATA.get(w,w))
            return " ".join(normalized)
        
        def ResellerProductFilter(self, OFFICIAL_PRODUCT_LIST,text):
            if pd.isna(text):
                return None
            
            text = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", text)

            text = text.lower()
            
            #Extract quantity from data name product
            qty_pattern = r'(\d+(?:\.\d+)?)(?:\s*x\s*\d+)?\s*(?:mg|g|gram|gr|kg|ml|l|caps|tablet|tabs|sachet|pcs)?'
            qty_remove_pattern = r'\d+(?:\.\d+)?(?:\s*x\s*\d+)?\s*(?:mg|g|gram|gr|kg|ml|l|caps|tablet|tabs|sachet|pcs)?'
            
            quantities = re.findall(qty_pattern, text)

            #Take numbers
            numeric_values = []
            for qty in quantities:
                try:
                    numeric_values.append(float(qty))
                except:
                    pass

            qty_final = ""
            if numeric_values:
                max_qty = int(max(numeric_values))
                qty_final = str(max_qty)

            #Erase qty from text for product search
            text_clean = re.sub(qty_remove_pattern,'',text).strip()
            
            official_product_dictio = {
                product : product.lower().split() for product in OFFICIAL_PRODUCT_LIST}
            
            for official_name, keyword in official_product_dictio.items():
                
                pattern = ''.join(f'(?=.*\\b{re.escape(word)}\\b)' for word in keyword)

                if re.search(pattern,text_clean):
                    if qty_final:
                        return f"{official_name} {qty_final}".strip()
                    else:
                        return official_name
            return f"{text_clean} {qty_final}".strip().title()
           
        def CleanStopwords(self,text,DATA_STOPWORDS):
            """
            Fungsi : Menghapus Stopword yang ada di dalam kalimat
            Args:
            1.text:str
            text = Input berisi kalimat yang mengandung StopWord
            2.DATA_STOPWORDS:List
            DATA_STOPWORDS adalah array/list yang berisi kata-kata stop words, bisa dikombinasikan dengan nltk.stopwords sesuai dengan bahasa masing-masing
            """
            #DATA_STOPWORDS = ['Ready Stock','Baru','Beli 1 Gratis 1', 'Beli 2 Gratis 1','Real 10% Advanced Niacinamide','Beli 2 Gratis 1','diskon','Terbatas','Paket Hemat 5 Pcs','order now, hemat 40k','new formula','flash sale','bpom original lokal', '100% lokal','Pre-Order','Toko Resmi','Live Sale','dengan','untuk','tampak','muda','keren','buat','muka','glowing','semua','dan','mencerahkan','kusam','jerawat','anti','melembabkan','pencerah','wajah','pelembab','perawatan','solusi','khusus','mencegah','tidak','ready','siap kirim','membandel','membantu','2 minggu','aman','susah putih','bruntusan','COD','terbaru','BPOM','cepat','sensitif','ampuh','memudarkan','lengkap','kemasan','free','free gift','Pembersih']
            text = str(text).lower()

            #hapus karakter khusus
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

            for stop in sorted(DATA_STOPWORDS, key=len,reverse = True):
                pattern = r"\b" + re.escape(stop.lower()) + r"\b"
                text = re.sub(pattern," ",text)

            #Normalize whitespace
            text = re.sub(r"\s+"," ",text).strip()

            return text
        
        def normalize_product_name(self,product_name, mapping_dict,category):
            """
            Return Canonical Name of product data in Reseller
            Input
            1.product_name:type(str) --Input of product name in dataframe
            2.mapping_dict:type(Dict) or JSON file -- [REQUIRED : NESTED DICTIONARY] Key,Value pairs of dictionary prodyuct name and values of keywords
            3.category:type(str) --Category product with string valued data as an input for nested mapping_dict
            """
            name=product_name.lower()

            #Ensure all category are available in mapping dictionary
            if category not in mapping_dict:
                return product_name
            
            category_dict = mapping_dict[category]

            for canonical_name,variants in category_dict.items():
                for v in variants:
                    if v.lower() in name:
                        return canonical_name.title()
            return product_name.title()
        
        def SlangWordFormatter(self,text, StandardDict:dict):
            """
            Fungsi untuk mengubah kata-kata slang menjadi kata-kata baku dalam Bahasa Indonesia

            Args:
            1.text:str
            text adalah input berisi kalimat dalam Bahasa Indonesia
            2.StandardDict:dict
            StandardDict adalah kamus yang berisi perbandingan dalam bentuk key,value pair utk mengubah kata 
            slang menjadi kata baku

            Returns
            text:str
            Teks yang dikembalikan dalam bentuk kalimat baku
            """
            pattern = re.compile(r'\b(' + '|'.join(map(re.escape,StandardDict.keys()))+ r')\b')

            return pattern.sub(lambda x:StandardDict[x.group(0)],text)
        def combine_negation(self, text):
            """
            Menggabungkan kata "Tidak" dan kata apapun yang ada setelahnya untuk mempertahankan
            konteks
            """
            combined_tokens = []
            skip_next = False
            tokens = text.split()

            for i in range(len(tokens)):
                if skip_next:
                    skip_next = False
                    continue

                if tokens[i].lower() == "tidak" and i + 1 < len(tokens):
                    combined_tokens.append(f"tidak_{tokens[i+1]}")
                    skip_next = True
                else:
                    combined_tokens.append(tokens[i])

            return " ".join(combined_tokens)
        
        def find_best_match(self,name, choices, threshold = 90):
            match, score,_ = process.extractOne(name, choices, scorer = fuzz.token_sort_ratio)
            if score >= threshold:
                return match
            else:
                return name
                

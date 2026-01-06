import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_fscore_support, confusion_matrix
# 1. Dizinler
DATA_DIR = r'C:\Users\Abdulvahit\Documents\Python\ubg\Warwick_QU_Dataset'
OUTPUT_DIR = os.path.join(DATA_DIR, 'analiz_sonuclari')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Yardımcı Fonksiyonlar

def largest_contour(contours):
    """En büyük alanlı contouru döndürür."""
    if not contours:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    return contours[idx]

# Görüntü İşleme ve Özellik Çıkarımı (analyze_glands & compute_shape_features) ile yapıldı.
# her bir görüntüdeki bezleri tek tek bulur ve fiziksel özelliklerini ölçmektedir.
# Ön İşleme: Maske dosyaları okunur, gürültüleri temizlemek için Morfolojik Açma/Kapama (Opening/Closing) işlemleri uygulanır.
# CCL (Connected Component Labeling): Maskedeki birbirine bağlı her bir bez yapısı ayrı bir nesne olarak etiketlenir.
# Şekil Özellikleri: Her bir bez için şu teknik metrikler hesaplanır:Solidity, Circularity,Eccentricity, Hu Moments
# Hu Moments: Şeklin döndürülse veya büyütülse bile değişmeyen matematiksel kimliği.

def compute_shape_features(cnt):
    """Contour üzerinden şekil özelliklerini hesaplar."""
    # Alan ve çevre
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Dışbükey kılıf (convex hull)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    solidity = (area / hull_area) if hull_area > 0 else 0.0
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    convexity = (perimeter / hull_perimeter) if hull_perimeter > 0 else 0.0  # 1'e yakınsa düzgün

    # Eksen hizalı bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = (w / h) if h > 0 else 0.0
    extent = (area / (w * h)) if (w * h) > 0 else 0.0

    # Eşdeğer çap
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    # Elips uydurma ile eksantriklik (eccentricity)
    eccentricity = 0.0
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), angle = ellipse  # MA: major axis, ma: minor axis
        a = max(MA, ma) / 2.0
        b = min(MA, ma) / 2.0
        if a > 0:
            eccentricity = np.sqrt(1 - (b**2 / a**2))

    # Hu momentleri (ölçek/rotasyon/öteleme invariant)
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()

    return {
        "Area": area,
        "Perimeter": perimeter,
        "HullArea": hull_area,
        "HullPerimeter": hull_perimeter,
        "Solidity": solidity,
        "Circularity": circularity,
        "Convexity": convexity,
        "AspectRatio": aspect_ratio,
        "Extent": extent,
        "EqDiameter": equivalent_diameter,
        "Eccentricity": eccentricity,
        "Hu1": hu[0], "Hu2": hu[1], "Hu3": hu[2],
        "Hu4": hu[3], "Hu5": hu[4], "Hu6": hu[5], "Hu7": hu[6],
        "BBoxX": x, "BBoxY": y, "BBoxW": w, "BBoxH": h,
    }

# Kural Tabanlı Sınıflandırma (classify_with_rules) yapılmıştır.
# Eğer bir bez çok uzamışsa (eccentricity > 0.80) doğrudan Anomali kabul edilir.
# Eğer bez yeterince katı (solidity > 0.88) ve yuvarlaksa (circularity > 0.40) Normal kabul edilir.
# kriterlere uymayanlar yine Anomali olarak işaretlenir.

def classify_with_rules(features, s_thr=0.88, c_thr=0.40, e_thr=0.80):
    """
    Basit kural tabanlı sınıflandırma:
    - Yüksek solidity ve yeterli circularity => Normal
    - Çok yüksek eksantriklik (uzamış/ince) => Anomali
    """
    s = features["Solidity"]
    c = features["Circularity"]
    e = features["Eccentricity"]

    # Önce çok uzamış yapıları anomalilere eklendi
    if e > e_thr:
        return "Anomali"

    if s > s_thr and c > c_thr:
        return "Normal"
    return "Anomali"

def analyze_glands(image_path, mask_path, area_min=500):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    if img is None or mask is None:
        return None, []

    # Maske ikilileştirme 
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # gürültü temizler
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # küçük delik kapatır

    # CCL
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    results = []
    overlay = img.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min:
            continue

        component_mask = (labels == i).astype(np.uint8) * 255
        # CHAIN_APPROX_NONE => daha doğru perimeter
        contours, hierarchy = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            continue

        cnt = largest_contour(contours)
        if cnt is None:
            continue

        feats = compute_shape_features(cnt)
        status = classify_with_rules(feats, s_thr=0.88, c_thr=0.40, e_thr=0.80)
        color = (0, 255, 0) if status == "Normal" else (0, 0, 255)

        # BBox çizimi + contour + hull
        x, y, w, h = feats["BBoxX"], feats["BBoxY"], feats["BBoxW"], feats["BBoxH"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.drawContours(overlay, [cnt], -1, color, 1)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(overlay, [hull], -1, (255, 165, 0), 1)  # turuncu hull

        # Etiket
        label_text = f"{status} | S:{feats['Solidity']:.2f} C:{feats['Circularity']:.2f}"
        cv2.putText(overlay, label_text, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        row = {
            "Dosya": os.path.basename(image_path),
            "ID": i,
            "Durum": status,
            **feats
        }
        results.append(row)

    return overlay, results


# 2. Çalıştırma

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"HATA: Yol bulunamadı: {DATA_DIR}")
    else:
        all_files = os.listdir(DATA_DIR)
        images = [f for f in all_files if f.lower().endswith('.bmp') and '_anno' not in f.lower()]
        print(f"Klasörde {len(images)} işlenebilir resim bulundu.")

        final_list = []
        for f in images:
            img_p = os.path.join(DATA_DIR, f)
            # Daha güvenli maske ismi türetme:
            base, ext = os.path.splitext(f)
            mask_name = f"{base}_anno.bmp"
            mask_p = os.path.join(DATA_DIR, mask_name)

            if os.path.exists(mask_p):
                res_img, res_data = analyze_glands(img_p, mask_p, area_min=500)
                if res_img is not None:
                    final_list.extend(res_data)
                    out_name = os.path.join(OUTPUT_DIR, f"analiz_{f}")
                    cv2.imwrite(out_name, res_img)
                    print(f"İşlendi: {f} -> {out_name}")
            else:
                print(f"Uyarı: Maske bulunamadı: {mask_p}")

        if final_list:
            df = pd.DataFrame(final_list)
            # Analiz raporunu yaz
            csv_path = os.path.join(DATA_DIR, 'Analiz_Raporu.csv')
            df.to_csv(csv_path, index=False)
            print("\nBitti! Rapor ve görseller kaydedildi:")
            print(f"- CSV: {csv_path}")
            print(f"- Görseller: {OUTPUT_DIR}")

            
            # EK: Görsel düzeyi benign/malign metrikler (Grade.csv ile)
            
            try:
                from sklearn.metrics import (
                    accuracy_score, precision_recall_fscore_support,
                    confusion_matrix, classification_report, roc_auc_score
                )

                # 1) Bileşen -> görüntü skor (p_anomali)
                df_pred = pd.read_csv(csv_path)
                df_pred['Stem'] = df_pred['Dosya'].str.replace('.bmp','', regex=False).str.strip().str.lower()
                df_scores = (
                    df_pred.groupby('Stem')['Durum']
                           .apply(lambda s: (s == 'Anomali').mean())
                           .reset_index(name='p_anomali')
                )

                # 2) Ground-truth (Grade.csv)
                grade_csv = os.path.join(DATA_DIR, 'Grade.csv')   # Grade.csv etiket dosyam
                if not os.path.exists(grade_csv):
                    print("Uyarı: Grade.csv bulunamadı; görsel düzeyi metrik atlandı.")
                else:
                    df_true = pd.read_csv(grade_csv)
                    df_true.columns = [c.strip() for c in df_true.columns]
                    df_true['Stem'] = df_true['name'].str.strip().str.lower()
                    df_true['GercekGorselEtiket'] = (
                        df_true['grade (GlaS)'].str.strip().str.lower()
                        .map({'benign':'Benign','malignant':'Malign'})
                    )

                    # 3) Birleştir
                    df_img = df_scores.merge(
                        df_true[['Stem','GercekGorselEtiket']],
                        on='Stem', how='inner'
                    )
                    if df_img.empty:
                        print("Uyarı: Stem eşleşmedi; dosya adlarını kontrol et.")
                    else:
                        # ROC-AUC (sürekli skor)
                        y_true_bin = (df_img['GercekGorselEtiket'].str.lower() == 'malign').astype(int)
                        y_score = df_img['p_anomali'].values
                        try:
                            auc = roc_auc_score(y_true_bin, y_score)
                        except Exception as e:
                            auc = None
                            print("AUC hesaplanamadı:", e)

                        # Grid-search ile en iyi tau (F1(Malign))
                        best = {'tau': None, 'f1': -1, 'acc': None, 'prec': None, 'rec': None}
                        for tau in np.linspace(0.30, 0.70, 9):
                            y_pred = np.where(df_img['p_anomali'] >= tau, 'Malign', 'Benign')
                            acc = accuracy_score(df_img['GercekGorselEtiket'], y_pred)
                            prec, rec, f1, _ = precision_recall_fscore_support(
                                df_img['GercekGorselEtiket'], y_pred, average='binary', pos_label='Malign'
                            )
                            if f1 > best['f1']:
                                best.update({'tau': tau, 'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec})

                        # Nihai rapor
                        y_pred_best = np.where(df_img['p_anomali'] >= best['tau'], 'Malign', 'Benign')
                        cm = confusion_matrix(df_img['GercekGorselEtiket'], y_pred_best, labels=['Benign','Malign'])

                        print("\n--- Görsel Düzeyi Performans (GlaS) ---")
                        if auc is not None:
                            print(f"ROC-AUC (p_anomali skoruyla): {auc:.3f}")
                        else:
                            print("ROC-AUC hesaplanamadı.")
                        print(f"En iyi eşik (tau): {best['tau']:.2f}")
                        print(f"Accuracy: {best['acc']:.3f}  Precision(Malign): {best['prec']:.3f}  Recall(Malign): {best['rec']:.3f}  F1(Malign): {best['f1']:.3f}")
                        print("Confusion Matrix [rows: True, cols: Pred]")
                        print(pd.DataFrame(cm, index=['True_Benign','True_Malign'], columns=['Pred_Benign','Pred_Malign']))

                        print("\nDetaylı Rapor:\n", classification_report(
                            df_img['GercekGorselEtiket'], y_pred_best, target_names=['Benign','Malign']
                        ))

                        # Kaydet: skorlar ve tahminler
                        out_csv = os.path.join(DATA_DIR, 'Gorsel_Duzeyi_Skorlar.csv')
                        df_save = df_img.copy()
                        df_save['PredBestTau'] = y_pred_best
                        df_save['BestTau'] = best['tau']
                        df_save.to_csv(out_csv, index=False)
                        print(f"\nGörsel düzeyi skor ve tahminler kaydedildi: {out_csv}")

                        # Metin rapor
                        perf_txt = os.path.join(DATA_DIR, 'Gorsel_Performans_Raporu.txt')
                        with open(perf_txt, 'w', encoding='utf-8') as f:
                            f.write("--- Görsel Düzeyi Performans (GlaS) ---\n")
                            if auc is not None:
                                f.write(f"ROC-AUC: {auc:.4f}\n")
                            f.write(f"Best tau: {best['tau']:.2f}\n")
                            f.write(f"Accuracy: {best['acc']:.4f}\nPrecision(Malign): {best['prec']:.4f}\nRecall(Malign): {best['rec']:.4f}\nF1(Malign): {best['f1']:.4f}\n\n")
                            f.write("Confusion Matrix [rows: True, cols: Pred]\n")
                            f.write(pd.DataFrame(cm, index=['True_Benign','True_Malign'],
                                                 columns=['Pred_Benign','Pred_Malign']).to_string())
                            f.write("\n\n")
                            f.write(classification_report(
                                df_img['GercekGorselEtiket'], y_pred_best, target_names=['Benign','Malign']
                            ))
                        print(f"Görsel düzeyi performans raporu kaydedildi: {perf_txt}")

            except ImportError:
                print("\nNot: scikit-learn bulunamadı. 'pip install scikit-learn' ile kurarak görsel düzeyi metrikleri çalıştırabilir.")
            except Exception as e:
                print("Görsel düzeyi değerlendirme aşamasında bir sorun oluştu:", e)

 

# Grafik ayarları
plt.rcParams['figure.dpi'] = 140
plt.rcParams['font.size'] = 10

DATA_DIR = r'C:\Users\Abdulvahit\Documents\Python\ubg\Warwick_QU_Dataset'
OUTPUT_DIR = os.path.join(DATA_DIR, 'analiz_sonuclari')
os.makedirs(OUTPUT_DIR, exist_ok=True)

analiz_csv = os.path.join(DATA_DIR, 'Analiz_Raporu.csv')
grade_csv = os.path.join(DATA_DIR, 'Grade.csv')

# 1) Verileri oku ve görsel düzeyi skorları oluştur
if not (os.path.exists(analiz_csv) and os.path.exists(grade_csv)):
    print('Gerekli dosyalar bulunamadı. Analiz_Raporu.csv veya Grade.csv eksik.')
else:
    df_pred = pd.read_csv(analiz_csv)
    df_pred['Stem'] = df_pred['Dosya'].str.replace('.bmp','', regex=False).str.strip().str.lower()
    df_scores = (
        df_pred.groupby('Stem')['Durum']
               .apply(lambda s: (s == 'Anomali').mean())
               .reset_index(name='p_anomali')
    )

    df_true = pd.read_csv(grade_csv)
    df_true.columns = [c.strip() for c in df_true.columns]
    df_true['Stem'] = df_true['name'].str.strip().str.lower()
    df_true['GercekGorselEtiket'] = (
        df_true['grade (GlaS)'].str.strip().str.lower().map({'benign':'Benign','malignant':'Malign'})
    )

    df_img = df_scores.merge(df_true[['Stem','GercekGorselEtiket']], on='Stem', how='inner')

    # 2) ROC eğrisi ve AUC
    y_true_bin = (df_img['GercekGorselEtiket'].str.lower() == 'malign').astype(int).values
    y_score = df_img['p_anomali'].values
    fpr, tpr, thr = roc_curve(y_true_bin, y_score)
    roc_auc = sk_auc(fpr, tpr)

    fig1 = plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1], color='navy', lw=1, ls='--')
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title('Görsel Düzeyi ROC Eğrisi (p_anomali)')
    plt.legend(loc='lower right')
    out_roc = os.path.join(OUTPUT_DIR, 'roc_curve_gorsel_duzey.png')
    plt.tight_layout(); plt.savefig(out_roc); plt.close(fig1)

    # 3) Eşik (tau) taraması: Accuracy / Precision / Recall / F1
    taus = np.linspace(0.30, 0.70, 21)
    accs, precs, recs, f1s = [], [], [], []
    for tau in taus:
        y_pred = np.where(df_img['p_anomali'] >= tau, 'Malign', 'Benign')
        acc = (y_pred == df_img['GercekGorselEtiket']).mean()
        prec, rec, f1, _ = precision_recall_fscore_support(
            df_img['GercekGorselEtiket'], y_pred, average='binary', pos_label='Malign'
        )
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    best_idx = int(np.argmax(f1s))
    best_tau = float(taus[best_idx])

    fig2 = plt.figure(figsize=(6,4))
    plt.plot(taus, accs, label='Accuracy', lw=2)
    plt.plot(taus, precs, label='Precision (Malign)', lw=2)
    plt.plot(taus, recs, label='Recall (Malign)', lw=2)
    plt.plot(taus, f1s, label='F1 (Malign)', lw=2)
    plt.axvline(best_tau, color='red', ls='--', lw=1, label=f'En iyi tau={best_tau:.2f}')
    plt.xlabel('Eşik (tau)'); plt.ylabel('Skor'); plt.title('Eşik Taraması: Görsel Düzeyi Metrikler')
    plt.legend(loc='best')
    out_tau = os.path.join(OUTPUT_DIR, 'tau_sweep_metrics.png')
    plt.tight_layout(); plt.savefig(out_tau); plt.close(fig2)

    # 4) En iyi tau ile Confusion Matrix (ısı haritası)
    y_pred_best = np.where(df_img['p_anomali'] >= best_tau, 'Malign', 'Benign')
    labels = ['Benign','Malign']
    cm = confusion_matrix(df_img['GercekGorselEtiket'], y_pred_best, labels=labels)

    fig3 = plt.figure(figsize=(4.5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Karmaşıklık Matrisi (tau={best_tau:.2f})')
    plt.colorbar(); tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels); plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('Gerçek'); plt.xlabel('Tahmin')
    out_cm = os.path.join(OUTPUT_DIR, 'confusion_matrix_best_tau.png')
    plt.tight_layout(); plt.savefig(out_cm); plt.close(fig3)

    # 5) p_anomali dağılımı: Benign vs Malign
    fig4 = plt.figure(figsize=(5.5,4))
    benign_scores = df_img.loc[df_img['GercekGorselEtiket']=='Benign','p_anomali']
    malign_scores = df_img.loc[df_img['GercekGorselEtiket']=='Malign','p_anomali']
    bins = np.linspace(0, 1, 21)
    plt.hist(benign_scores, bins=bins, alpha=0.6, label='Benign', color='green')
    plt.hist(malign_scores, bins=bins, alpha=0.6, label='Malign', color='red')
    plt.axvline(best_tau, color='black', ls='--', lw=1, label=f'tau={best_tau:.2f}')
    plt.xlabel('p_anomali (Anomali bileşen oranı)')
    plt.ylabel('Frekans'); plt.title('p_anomali Dağılımı (Görsel Düzeyi)')
    plt.legend()
    out_hist = os.path.join(OUTPUT_DIR, 'p_anomali_histogram.png')
    plt.tight_layout(); plt.savefig(out_hist); plt.close(fig4)

    print('Grafikler kaydedildi:')
    print(' -', out_roc)
    print(' -', out_tau)
    print(' -', out_cm)
    print(' -', out_hist)
    print(f'En iyi tau: {best_tau:.2f}  AUC: {roc_auc:.3f}')
           
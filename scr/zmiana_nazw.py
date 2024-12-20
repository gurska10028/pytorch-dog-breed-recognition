# skrypt, który zmienia recznie nazwy zdjec

import os

def zmien_nazwy_zdjec(folder, nowa_nazwa_podstawowa): #nowa_nazwa_podstawowa - podstawa do której będzie dodawane _1, _2 itd.
    sciezki_plikow = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for i, sciezka in enumerate(sciezki_plikow, start=1):
        nowa_nazwa = f"{nowa_nazwa_podstawowa}_{i}.jpg"
        nowa_sciezka = os.path.join(folder, nowa_nazwa)
        os.rename(sciezka, nowa_sciezka)
        print(f"Zmieniono nazwę: {sciezka} -> {nowa_sciezka}")

    print("Zmiana nazw zakończona.")

folder_zdjec = r'C:\Users\amelk\Documents\Erasmus_3_semestr\projekt MN\Project\data\obrazy'  
zmien_nazwy_zdjec(folder_zdjec, "zdjecie")

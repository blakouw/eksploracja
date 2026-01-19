# Jump Counter Wrapper

Jump Counter Wrapper to aplikacja React, która działa jako wrapper do TensorFlow i Mediapipe Pose. Aplikacja pozwala na wykrywanie ruchu użytkownika przez kamerę i liczenie skoków w czasie rzeczywistym.

Aplikacja analizuje obraz z kamery, wykrywa pozycję ciała użytkownika i rysuje szkielet na ekranie. Liczba wykonanych skoków jest wyświetlana w czasie rzeczywistym.

Projekt wykorzystuje React jako frontend, TensorFlow.js do detekcji ciała, model MoveNet do wykrywania pozycji ciała, Mediapipe Pose do rysowania szkieletu oraz komponent Webcam do obsługi kamery w przeglądarce.

Model MoveNet analizuje obraz z kamery co kilkadziesiąt milisekund. Funkcja drawSkeleton rysuje szkielet na wideo, a funkcja countJumps liczy skoki i aktualizuje wynik.

Struktura projektu jest prosta. App.tsx zawiera główny komponent z logiką detekcji i liczenia skoków, App.style.ts definiuje style komponentów, a utlis.ts zawiera funkcje pomocnicze takie jak drawSkeleton i countJumps.


```npm install```


```npm start```


Aplikacja otworzy się w przeglądarce z podglądem kamery i licznikiem skoków.

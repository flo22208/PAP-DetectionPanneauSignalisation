from ultralytics import YOLO

def main():
    # Charger le modèle entraîné (remplacez 'best.pt' par le chemin de votre modèle si besoin)
    model = YOLO("runs/detect/train_nano_2000/weights/best.pt")
    
    # Évaluer le modèle sur le jeu de test défini dans dataset.yaml
    metrics = model.val(data="PAP-DetectionPanneauSignalisation/dataset.yaml", split='test')    

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Implementación de la pérdida contrastiva para aprendizaje de tripletes.
    """
    def __init__(self, temperature=0.07, margin=0.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, sim_pos, sim_neg):
        """
        Calcula la pérdida contrastiva.
        
        Args:
            sim_pos: Similitud entre pares positivos [batch_size]
            sim_neg: Similitud entre pares negativos [batch_size]
            
        Returns:
            Pérdida escalar
        """
        # Aplicar margen opcional
        sim_neg = sim_neg - self.margin
        
        # Escalamiento con temperatura
        sim_pos = sim_pos / self.temperature
        sim_neg = sim_neg / self.temperature
        
        # Implementación de la pérdida contrastiva con InfoNCE
        numerator = torch.exp(sim_pos)
        denominator = numerator + torch.exp(sim_neg)
        loss = -torch.log(numerator / denominator)
        
        return loss.mean()

class MultimodalContrastiveLoss(nn.Module):
    """
    Pérdida contrastiva combinando imágenes y texto.
    """
    def __init__(self, temperature=0.07, alpha=0.5):
        super(MultimodalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # Ponderación entre pérdida de imagen y texto
        self.contrastive_loss = ContrastiveLoss(temperature)
        
    def forward(self, outputs):
        """
        Calcula la pérdida contrastiva multimodal.
        
        Args:
            outputs: Diccionario con embeddings y similitudes
            
        Returns:
            Pérdida escalar y métricas adicionales
        """
        # Pérdida principal (ya combina imagen y texto si están disponibles)
        main_loss = self.contrastive_loss(
            outputs['similarities']['positive'],
            outputs['similarities']['negative']
        )
        
        # Calcular accuracy de triplete (cuántas veces sim_pos > sim_neg)
        triplet_acc = (outputs['similarities']['positive'] > outputs['similarities']['negative']).float().mean()
        
        result = {
            'total_loss': main_loss,
            'triplet_accuracy': triplet_acc
        }
        
        # Si hay embeddings de texto separados, podemos calcular pérdidas adicionales
        if 'text' in outputs['embeddings'] and outputs['embeddings']['text'] is not None:
            # Similitudes entre embeddings de texto
            text_sim_pos = F.cosine_similarity(
                outputs['embeddings']['text']['original'], 
                outputs['embeddings']['text']['generated'], 
                dim=1
            )
            text_sim_neg = F.cosine_similarity(
                outputs['embeddings']['text']['original'], 
                outputs['embeddings']['text']['negative'], 
                dim=1
            )
            
            # Pérdida para texto
            text_loss = self.contrastive_loss(text_sim_pos, text_sim_neg)
            
            # Similitudes cruzadas (imagen-texto)
            cross_sim_pos = F.cosine_similarity(
                outputs['embeddings']['original'], 
                outputs['embeddings']['text']['generated'], 
                dim=1
            )
            cross_sim_neg = F.cosine_similarity(
                outputs['embeddings']['original'], 
                outputs['embeddings']['text']['negative'], 
                dim=1
            )
            
            # Pérdida cruzada
            cross_loss = self.contrastive_loss(cross_sim_pos, cross_sim_neg)
            
            # Accuracy de triplete para texto y cruzada
            text_acc = (text_sim_pos > text_sim_neg).float().mean()
            cross_acc = (cross_sim_pos > cross_sim_neg).float().mean()
            
            # Pérdida total con componentes adicionales
            total_loss = main_loss + self.alpha * (text_loss + cross_loss)
            
            result.update({
                'total_loss': total_loss,
                'image_loss': main_loss,
                'text_loss': text_loss,
                'cross_loss': cross_loss,
                'text_accuracy': text_acc,
                'cross_accuracy': cross_acc
            })
        
        return result
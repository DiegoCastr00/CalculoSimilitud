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
        # Escalamiento con temperatura
        sim_pos = sim_pos / self.temperature
        sim_neg = sim_neg / self.temperature
        
        # Implementación de la pérdida contrastiva con InfoNCE
        # -log(exp(sim_pos/τ) / (exp(sim_pos/τ) + exp(sim_neg/τ)))
        numerator = torch.exp(sim_pos)
        denominator = numerator + torch.exp(sim_neg)
        loss = -torch.log(numerator / denominator)
        
        return loss.mean()

class TripletLoss(nn.Module):
    """
    Implementación alternativa de la pérdida de triplete.
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Calcula la pérdida de triplete.
        
        Args:
            anchor: Embeddings ancla [batch_size, embed_dim]
            positive: Embeddings positivos [batch_size, embed_dim]
            negative: Embeddings negativos [batch_size, embed_dim]
            
        Returns:
            Pérdida escalar
        """
        distance_pos = (anchor - positive).pow(2).sum(1)
        distance_neg = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_pos - distance_neg + self.margin)
        
        return losses.mean()

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
            outputs: Diccionario con 'similarities' y posiblemente 'text_similarities'
            
        Returns:
            Pérdida escalar
        """
        # Pérdida de imagen
        image_loss = self.contrastive_loss(
            outputs['similarities']['positive'],
            outputs['similarities']['negative']
        )
        
        # Si hay embeddings de texto
        if 'text' in outputs['embeddings'] and outputs['embeddings']['text'] is not None:
            # Calcular similitudes texto-imagen
            original_text = outputs['embeddings']['text']['original']
            original_image = outputs['embeddings']['original']
            
            # Similitud texto-imagen para pares positivos (original-original)
            text_image_sim = F.cosine_similarity(original_text, original_image, dim=1)
            
            # Similitud texto-imagen para pares negativos (original-negative)
            negative_image = outputs['embeddings']['negative']
            text_neg_sim = F.cosine_similarity(original_text, negative_image, dim=1)
            
            # Pérdida texto-imagen
            text_loss = self.contrastive_loss(text_image_sim, text_neg_sim)
            
            # Combinación ponderada
            total_loss = self.alpha * image_loss + (1 - self.alpha) * text_loss
            
            return {
                'total_loss': total_loss,
                'image_loss': image_loss,
                'text_loss': text_loss
            }
        
        return {
            'total_loss': image_loss,
            'image_loss': image_loss
        }
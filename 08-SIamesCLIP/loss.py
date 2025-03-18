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
        
    def forward(self, sim_pos, sim_neg, temperature=None):
        """
        Calcula la pérdida contrastiva.
        
        Args:
            sim_pos: Similitud entre pares positivos [batch_size]
            sim_neg: Similitud entre pares negativos [batch_size]
            temperature: Temperatura opcional (sobreescribe la predeterminada)
            
        Returns:
            Pérdida escalar
        """
        # Usar temperatura proporcionada o la predeterminada
        temp = temperature if temperature is not None else self.temperature
        
        # Aplicar margen opcional
        sim_neg = sim_neg - self.margin
        
        # Escalamiento con temperatura
        sim_pos = sim_pos / temp
        sim_neg = sim_neg / temp
        
        # Implementación de la pérdida contrastiva con InfoNCE
        numerator = torch.exp(sim_pos)
        denominator = numerator + torch.exp(sim_neg)
        loss = -torch.log(numerator / denominator)
        
        return loss.mean()

class MultimodalContrastiveLoss(nn.Module):
    """
    Pérdida contrastiva combinando imágenes y texto.
    """
    def __init__(self, temperature=0.07, alpha_text=0.3, alpha_cross=0.3):
        super(MultimodalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.alpha_text = alpha_text    # Ponderación para pérdida de texto
        self.alpha_cross = alpha_cross  # Ponderación para pérdida cruzada
        self.contrastive_loss = ContrastiveLoss(temperature)
        
    def forward(self, outputs):
        """
        Calcula la pérdida contrastiva multimodal.
        
        Args:
            outputs: Diccionario con embeddings y similitudes
            
        Returns:
            Pérdida escalar y métricas adicionales
        """
        # Verificar que el diccionario de salida tenga la estructura esperada
        if not isinstance(outputs, dict) or 'similarities' not in outputs or 'embeddings' not in outputs:
            raise ValueError("El output del modelo debe ser un diccionario con claves 'similarities' y 'embeddings'")
            
        if not isinstance(outputs['similarities'], dict) or 'positive' not in outputs['similarities'] or 'negative' not in outputs['similarities']:
            raise ValueError("outputs['similarities'] debe ser un diccionario con claves 'positive' y 'negative'")
            
        # Extraer temperatura dinámica si está disponible en el modelo
        temperature = outputs.get('temperature', self.temperature)
        
        # Pérdida principal (ya combina imagen y texto si están disponibles)
        main_loss = self.contrastive_loss(
            outputs['similarities']['positive'],
            outputs['similarities']['negative'],
            temperature=temperature
        )
        
        # Calcular accuracy de triplete (cuántas veces sim_pos > sim_neg)
        triplet_acc = (outputs['similarities']['positive'] > outputs['similarities']['negative']).float().mean()
        
        result = {
            'total_loss': main_loss,
            'image_loss': main_loss,  # Para consistencia en logs
            'triplet_accuracy': triplet_acc
        }
        
        # Validación más estricta para los embeddings de texto antes de calcular pérdidas adicionales
        has_valid_text_embeddings = (
            'text' in outputs['embeddings'] and 
            outputs['embeddings']['text'] is not None and
            isinstance(outputs['embeddings']['text'], dict) and
            'original' in outputs['embeddings']['text'] and
            'generated' in outputs['embeddings']['text'] and
            'negative' in outputs['embeddings']['text'] and
            outputs['embeddings']['text']['original'] is not None and
            outputs['embeddings']['text']['generated'] is not None and
            outputs['embeddings']['text']['negative'] is not None
        )
        
        # Si hay embeddings de texto válidos, podemos calcular pérdidas adicionales
        if has_valid_text_embeddings:
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
            text_loss = self.contrastive_loss(text_sim_pos, text_sim_neg, temperature=temperature)
            
            # Similitudes cruzadas (imagen-texto)
            i2t_sim_pos = F.cosine_similarity(
                outputs['embeddings']['original'], 
                outputs['embeddings']['text']['generated'], 
                dim=1
            )
            i2t_sim_neg = F.cosine_similarity(
                outputs['embeddings']['original'], 
                outputs['embeddings']['text']['negative'], 
                dim=1
            )
            
            # Similitudes cruzadas (texto-imagen) - añadimos este componente
            t2i_sim_pos = F.cosine_similarity(
                outputs['embeddings']['text']['original'], 
                outputs['embeddings']['generated'], 
                dim=1
            )
            t2i_sim_neg = F.cosine_similarity(
                outputs['embeddings']['text']['original'], 
                outputs['embeddings']['negative'], 
                dim=1
            )
            
            # Pérdidas cruzadas
            i2t_loss = self.contrastive_loss(i2t_sim_pos, i2t_sim_neg, temperature=temperature)
            t2i_loss = self.contrastive_loss(t2i_sim_pos, t2i_sim_neg, temperature=temperature)
            cross_loss = (i2t_loss + t2i_loss) / 2  # Promedio de ambas direcciones
            
            # Accuracies de triplete para texto y cruzadas
            text_acc = (text_sim_pos > text_sim_neg).float().mean()
            i2t_acc = (i2t_sim_pos > i2t_sim_neg).float().mean()
            t2i_acc = (t2i_sim_pos > t2i_sim_neg).float().mean()
            cross_acc = (i2t_acc + t2i_acc) / 2  # Promedio de ambas direcciones
            
            # Pérdida total con componentes ponderados separadamente
            total_loss = main_loss + self.alpha_text * text_loss + self.alpha_cross * cross_loss
            
            result.update({
                'total_loss': total_loss,
                'text_loss': text_loss,
                'i2t_loss': i2t_loss,
                't2i_loss': t2i_loss,
                'cross_loss': cross_loss,
                'text_accuracy': text_acc,
                'i2t_accuracy': i2t_acc,
                't2i_accuracy': t2i_acc,
                'cross_accuracy': cross_acc
            })
        
        return result
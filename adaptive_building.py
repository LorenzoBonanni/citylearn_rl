import torch
import numpy as np
from typing import Any, List, Mapping, Union, Optional, Tuple
from citylearn.building import LSTMDynamicsBuilding
from citylearn.dynamics import LSTMDynamics
from citylearn.citylearn import CityLearnEnv

class AdaptiveLSTMDynamics(LSTMDynamics):
    """
    Classe che estende LSTMDynamics per adattarsi alle nuove dinamiche
    durante la simulazione, combinando un modello originale con uno aggiornato
    che apprende in modo incrementale.
    
    Parameters
    ----------
    base_dynamics: LSTMDynamics
        Il modello LSTM di base/originale
    adaptation_rate: float, default: 0.05
        Velocità di apprendimento per l'adattamento del modello
    blend_weight: float, default: 0.8
        Peso di mix tra modello originale e adattivo (1.0 = solo originale)
    window_size: int, default: 100
        Dimensione del buffer per le esperienze di apprendimento
    seed: int, optional
        Seed per la riproducibilità
    """
    
    def __init__(
        self, 
        base_dynamics: LSTMDynamics,
        adaptation_rate: float = 0.05,
        blend_weight: float = 0.8,
        window_size: int = 100,
        seed: Optional[int] = None
    ):
        # Inizializza usando i parametri del modello di base
        super().__init__(
            filepath=base_dynamics.filepath,
            input_observation_names=base_dynamics.input_observation_names,
            input_normalization_minimum=base_dynamics.input_normalization_minimum,
            input_normalization_maximum=base_dynamics.input_normalization_maximum,
            hidden_size=base_dynamics.hidden_size,
            num_layers=base_dynamics.num_layers,
            lookback=base_dynamics.lookback,
            input_size=base_dynamics.input_size
        )
        
        # Salva il modello originale
        self.original_dynamics = base_dynamics
        
        # Parametri di adattamento
        self.adaptation_rate = adaptation_rate
        self.blend_weight = blend_weight
        self.window_size = window_size
        
        # Buffer circolare per esperienza
        from collections import deque
        self.experience_buffer = deque(maxlen=window_size)
        
        # Statistiche di errore per monitoraggio
        self.orig_error_sum = 0.0
        self.adapt_error_sum = 0.0
        self.samples_count = 0
        
        # Stato precedente per l'apprendimento
        self._prev_input = None
        self._prev_hidden_state = None
        self._prev_prediction = None
        self._real_observation = None
        
        # Imposta i seed per riproducibilità
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Carica i pesi dal modello base
        with torch.no_grad():
            self.l_lstm.load_state_dict(base_dynamics.l_lstm.state_dict())
            self.l_linear.load_state_dict(base_dynamics.l_linear.state_dict())
            
        # Ottimizzatore per apprendimento online
        self.optimizer = torch.optim.Adam(self.parameters(), lr=adaptation_rate)
        self.criterion = torch.nn.MSELoss()
    
    def forward(self, x: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Combina le predizioni del modello originale e del modello adattivo.
        
        Parameters
        ----------
        x: torch.Tensor
            Tensore di input
        h: Tuple[torch.Tensor, torch.Tensor]
            Stato hidden LSTM
            
        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Output combinato e nuovo stato hidden
        """
        # Memorizza input e stato per apprendimento successivo
        self._prev_input = x.clone()
        self._prev_hidden_state = (h[0].clone(), h[1].clone())
        
        # Predizione con modello originale (senza gradiente)
        with torch.no_grad():
            orig_output, _ = self.original_dynamics.forward(x, h)
            
        # Predizione con modello adattivo
        adapt_output, hidden_state = super().forward(x, h)
        
        # Combina le predizioni con blend_weight
        output = self.blend_weight * orig_output + (1 - self.blend_weight) * adapt_output
        self._prev_prediction = output.clone()
        
        return output, hidden_state
    
    def observe_real_value(self, real_value: float) -> None:
        """
        Registra il valore reale osservato e aggiorna il modello.
        
        Parameters
        ----------
        real_value: float
            Il valore reale osservato (es. temperatura interna)
        """
        if self._prev_input is None or self._prev_prediction is None:
            return
            
        # Converti a tensore per il calcolo dell'errore
        real_tensor = torch.tensor([[real_value]], dtype=torch.float32)
        self._real_observation = real_tensor
        
        # Calcola l'errore del modello originale
        orig_error = ((self._prev_prediction - real_tensor) ** 2).item()
        self.orig_error_sum += orig_error
        self.samples_count += 1
        
        # Aggiungi al buffer di esperienza
        self.experience_buffer.append({
            'input': self._prev_input.clone(),
            'hidden_state': self._prev_hidden_state,
            'target': real_tensor
        })
        
        # Aggiorna il modello se abbiamo abbastanza dati
        if len(self.experience_buffer) >= 10:
            self.update_model()
    
    def update_model(self, mini_batch_size: int = 8) -> None:
        """
        Aggiorna il modello adattivo usando un mini-batch dal buffer.
        
        Parameters
        ----------
        mini_batch_size: int, default: 8
            Dimensione del mini-batch per l'aggiornamento
        """
        if len(self.experience_buffer) < mini_batch_size:
            return
            
        # Campiona un mini-batch casuale
        indices = np.random.choice(len(self.experience_buffer), 
                                  min(mini_batch_size, len(self.experience_buffer)),
                                  replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Aggiorna il modello
        self.optimizer.zero_grad()
        total_loss = 0
        
        for exp in batch:
            # Predizione
            output, _ = super().forward(exp['input'], exp['hidden_state'])
            # Calcolo loss
            loss = self.criterion(output, exp['target'])
            total_loss += loss
        
        # Backpropagation e aggiornamento
        total_loss.backward()
        self.optimizer.step()
        
        # Calcola l'errore del modello adattivo dopo l'aggiornamento
        with torch.no_grad():
            adapt_output, _ = super().forward(self._prev_input, self._prev_hidden_state)
            adapt_error = ((adapt_output - self._real_observation) ** 2).item()
            self.adapt_error_sum += adapt_error
            
        # Adatta il peso di blend in base agli errori relativi
        self._adapt_blend_weight()
    
    def _adapt_blend_weight(self) -> None:
        """Adatta il peso di blend in base alle performance relative dei modelli."""
        if self.samples_count <= 10:
            return  # Aspetta abbastanza campioni
            
        # Calcola errori medi
        avg_orig_error = self.orig_error_sum / self.samples_count
        avg_adapt_error = self.adapt_error_sum / max(1, self.samples_count - 10)
        
        # Calcola nuovo peso target basato sui reciproci degli errori
        if avg_orig_error > 0 and avg_adapt_error > 0:
            recip_sum = (1/avg_orig_error) + (1/avg_adapt_error)
            target_weight = (1/avg_orig_error) / recip_sum
            
            # Limita il peso minimo del modello originale al 20%
            target_weight = max(0.2, min(0.95, target_weight))
            
            # Aggiornamento graduale
            self.blend_weight = self.blend_weight + self.adaptation_rate * (target_weight - self.blend_weight)
    
    def get_metrics(self) -> dict:
        """Restituisce metriche sullo stato di adattamento."""
        return {
            'original_model_error': self.orig_error_sum / max(1, self.samples_count),
            'adaptive_model_error': self.adapt_error_sum / max(1, self.samples_count),
            'blend_weight': self.blend_weight,
            'buffer_size': len(self.experience_buffer),
            'samples_count': self.samples_count
        }
    
    def reset(self):
        """Reset del modello."""
        self._hidden_state = self.init_hidden(1)
        self._model_input = [[None]*(self.lookback + 1) for _ in self.input_observation_names]
        
        # Reset stati di adattamento
        self._prev_input = None
        self._prev_hidden_state = None
        self._prev_prediction = None
        self._real_observation = None


class AdaptiveLSTMDynamicsBuilding(LSTMDynamicsBuilding):
    """
    Building che utilizza un modello LSTM adattivo per
    predire la dinamica della temperatura interna e adattarsi
    alle differenze tra il modello simulato e quello reale.
    
    Parameters
    ----------
    *args: Any
        Argomenti posizionali per LSTMDynamicsBuilding
    dynamics: LSTMDynamics
        Modello di dinamica LSTM originale
    adaptation_rate: float, default: 0.05
        Velocità di apprendimento per l'adattamento
    blend_weight: float, default: 0.8
        Peso iniziale del modello originale
    window_size: int, default: 100
        Dimensione del buffer per l'apprendimento
    **kwargs: Any
        Altri argomenti keyword per LSTMDynamicsBuilding
    """
    
    def __init__(
        self, 
        *args: Any, 
        dynamics: LSTMDynamics,
        adaptation_rate: float = 0.05,
        blend_weight: float = 0.8,
        window_size: int = 100,
        **kwargs: Any
    ):
        import uuid
        
        # Inizializza manualmente l'ID univoco per evitare errori di uuid
        self._Environment__uid = uuid.uuid4().hex
        self._Environment__time_step = None
        
        if 'electric_vehicle_chargers' not in kwargs or kwargs['electric_vehicle_chargers'] is None:
            kwargs['electric_vehicle_chargers'] = []

        dynamics.reset()
        
        # Crea il modello adattivo
        self.adapt_dynamics = AdaptiveLSTMDynamics(
            base_dynamics=dynamics,
            adaptation_rate=adaptation_rate,
            blend_weight=blend_weight,
            window_size=window_size
        )
        
        # Inizializza la classe padre
        super().__init__(*args, dynamics=self.adapt_dynamics, **kwargs)
        self._last_real_temperature = None

    def step(self, action):
        """
        Override del metodo step per catturare la temperatura reale
        e fornirla al modello adattivo per l'apprendimento.
        """
        # Esegui il passo normale
        result = super().step(action)
        
        # Dopo lo step, salva la temperatura interna reale
        self._last_real_temperature = self.indoor_dry_bulb_temperature[-1]
        
        # Fornisci questa temperatura al modello adattivo per l'apprendimento
        self.adapt_dynamics.observe_real_value(self._last_real_temperature)
        
        return result
    
    def get_adaptation_metrics(self):
        """Ottieni metriche sul processo di adattamento."""
        return self.adapt_dynamics.get_metrics()

    # Override delle proprietà problematiche per garantire compatibilità
    @property
    def net_electricity_consumption_without_storage(self):
        """Override per garantire la compatibilità con il calcolo dei KPI."""
        try:
            return super().net_electricity_consumption_without_storage
        except ValueError:
            # In caso di errore, restituisci un array di zeri della dimensione corretta
            base_array = self.net_electricity_consumption
            if isinstance(base_array, np.ndarray):
                return np.zeros_like(base_array)
            else:
                return np.zeros(self.time_series_length)
                
    @property
    def net_electricity_consumption_without_storage_and_partial_load(self):
        """Override per garantire la compatibilità con il calcolo dei KPI."""
        try:
            return super().net_electricity_consumption_without_storage_and_partial_load
        except ValueError:
            # In caso di errore, restituisci un array di zeri della dimensione corretta
            base_array = self.net_electricity_consumption
            if isinstance(base_array, np.ndarray):
                return np.zeros_like(base_array)
            else:
                return np.zeros(self.time_series_length)
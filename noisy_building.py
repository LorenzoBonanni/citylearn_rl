import torch
import numpy as np
from typing import Any, List, Mapping, Union, Optional, Tuple
from citylearn.building import LSTMDynamicsBuilding
from citylearn.dynamics import LSTMDynamics
from citylearn.citylearn import CityLearnEnv

class NoisyLSTMDynamics(LSTMDynamics):
    """
    Classe che estende LSTMDynamics per aggiungere rumore ai dati di input,
    permettendo di "sporcare" i dati durante il training o l'inferenza.
    
    Parameters
    ----------
    base_dynamics: LSTMDynamics
        Il modello LSTM di base che viene esteso/modificato
    noise_level: float, default: 0.05
        Livello di rumore da aggiungere (deviazione standard del rumore gaussiano)
    apply_noise_to: list, optional
        Lista di nomi delle osservazioni a cui applicare il rumore.
        Se None, applica il rumore a tutte le osservazioni.
    noise_type: str, default: 'gaussian'
        Tipo di rumore da aggiungere ('gaussian', 'uniform')
    seed: int, optional
        Seed per la generazione del rumore, utile per rendere riproducibili gli esperimenti
    """
    
    def __init__(
        self, 
        base_dynamics: LSTMDynamics,
        noise_level: float = 0.05,
        noise_mean: float = 0.0,
        apply_noise_to: Optional[List[str]] = None,
        noise_type: str = 'gaussian',
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
        
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.noise_mean = noise_mean
        self.apply_noise_to = apply_noise_to or self.input_observation_names
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggiunge rumore al tensore di input.
        
        Parameters
        ----------
        x: torch.Tensor
            Tensore di input che può essere di forma [batch_size, features] (output LSTM)
            o [batch_size, seq_length, features] (input LSTM)
            
        Returns
        -------
        torch.Tensor
            Tensore con rumore aggiunto
        """
        if x.dim() == 2:
            noisy_x = x.clone()  # Create a copy to modify
        
            for i, obs_name in enumerate(self.input_observation_names):
                # Skip noise for attributes in the exclude list or 'hour'
                if obs_name not in self.apply_noise_to or obs_name == 'hour':
                    continue

                if self.noise_type == 'gaussian':
                    noise = torch.randn_like(x[:, i:i+1]) * self.noise_level + self.noise_mean
                elif self.noise_type == 'uniform':
                    noise = (torch.rand_like(x[:, i:i+1]) * 2 - 1) * self.noise_level + self.noise_mean
                else:
                    raise ValueError(f"Tipo di rumore '{self.noise_type}' non supportato")
                
                noisy_x[:, i:i+1] += noise
        noisy_x = noisy_x * 1 if np.random.random() > 0.5 else noisy_x * -1
        return noisy_x


    def forward(self, x: torch.Tensor, h: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        output, hidden_state = super().forward(x, h)
        output = torch.clamp(self.add_noise(output), min=-1.0, max=1.0)
        return output, hidden_state
    
    def reset(self):
        """Reset del modello"""
        try:
            # Carica il state dict del modello originale
            try:
                state_dict = torch.load(self.filepath)['model_state_dict']
            except RuntimeError:
                state_dict = torch.load(self.filepath, map_location=torch.device('cpu'))['model_state_dict']
            except:
                state_dict = torch.load(self.filepath)
            
            # Carica il state dict direttamente (senza prefissi problematici)
            self.load_state_dict(state_dict)
            
        except Exception as e:
            pass  # Silently handle errors
        
        # Inizializza hidden state e model input
        self._hidden_state = self.init_hidden(1)
        self._model_input = [[None]*(self.lookback + 1) for _ in self.input_observation_names]
    
    def set_noise_parameters(self, noise_level=None, noise_mean=None, apply_noise_to=None, noise_type=None):
        """ 
        Aggiorna i parametri del rumore.
        """
        if noise_level is not None:
            self.noise_level = noise_level
        if noise_mean is not None:
            self.noise_mean = noise_mean
        if apply_noise_to is not None:
            self.apply_noise_to = apply_noise_to
        if noise_type is not None:
            self.noise_type = noise_type

class NoisyLSTMDynamicsBuilding(LSTMDynamicsBuilding):
    """
    Building che utilizza un modello LSTM con rumore per
    predire la dinamica della temperatura interna con dati "sporcati".
    
    Parameters
    ----------
    *args: Any
        Argomenti posizionali per LSTMDynamicsBuilding
    dynamics: LSTMDynamics
        Modello di dinamica LSTM
    noise_level: float, default: 0.05
        Livello di rumore da aggiungere
    apply_noise_to: List[str], optional
        Lista di osservazioni a cui applicare il rumore
    noise_type: str, default: 'gaussian'
        Tipo di rumore da aggiungere ('gaussian', 'uniform')
    **kwargs: Any
        Altri argomenti keyword per LSTMDynamicsBuilding
    """
    
    def __init__(
        self, 
        *args: Any, 
        dynamics: LSTMDynamics,
        noise_level: float = 0.05,
        noise_mean: float = 0.0,
        apply_noise_to: Optional[List[str]] = None,
        noise_type: str = 'gaussian',
        **kwargs: Any
    ):
        if 'electric_vehicle_chargers' not in kwargs or kwargs['electric_vehicle_chargers'] is None:
            kwargs['electric_vehicle_chargers'] = []
            
        dynamics.reset()
        
        self.noisy_dynamics = NoisyLSTMDynamics(
            base_dynamics=dynamics,
            noise_level=noise_level,
            noise_mean=noise_mean,
            apply_noise_to=apply_noise_to,
            noise_type=noise_type
        )
        
        self.noisy_dynamics.reset()
            
        super().__init__(*args, dynamics=self.noisy_dynamics, **kwargs)
    
    @property 
    def simulate_dynamics(self) -> bool:
        """Whether to predict indoor dry-bulb temperature at current `time_step`."""
        base_simulate = not self.ignore_dynamics
        if base_simulate and hasattr(self.dynamics, '_model_input') and self.dynamics._model_input is not None:
            try:
                return self.dynamics._model_input[0][0] is not None
            except (IndexError, TypeError):
                return False
        return base_simulate
    
    def modify_noise(self, noise_level=None, noise_mean=None, apply_noise_to=None, noise_type=None):
        """
        Modifica i parametri del rumore durante la simulazione.
        """
        self.noisy_dynamics.set_noise_parameters(
            noise_level=noise_level,
            noise_mean=noise_mean,
            apply_noise_to=apply_noise_to,
            noise_type=noise_type
        )

    # def update_indoor_dry_bulb_temperature(self):
    #     """Predict and update indoor dry-bulb temperature for current `time_step` with explicit noise injection."""
    #     # predict
    #     model_input_tensor = torch.tensor(self.get_dynamics_input().T)
    #     model_input_tensor = model_input_tensor[np.newaxis, :, :]
    #     hidden_state = tuple([h.data for h in self.dynamics._hidden_state])
    #     indoor_dry_bulb_temperature_norm, self.dynamics._hidden_state = self.dynamics(model_input_tensor.float(), hidden_state)

    #     indoor_dry_bulb_temperature_norm_noisy = indoor_dry_bulb_temperature_norm.item() + add_noise
    #     # Clamp to valid normalized range
    #     indoor_dry_bulb_temperature_norm_noisy = np.clip(indoor_dry_bulb_temperature_norm_noisy, -1.0, 1.0)

    #     # update dry bulb temperature for current time step in model input
    #     ix = self.dynamics.input_observation_names.index('indoor_dry_bulb_temperature')
    #     self.dynamics._model_input[ix][-1] = indoor_dry_bulb_temperature_norm_noisy

    #     # unnormalize temperature
    #     low_limit, high_limit = self.dynamics.input_normalization_minimum[ix], self.dynamics.input_normalization_maximum[ix]
    #     indoor_dry_bulb_temperature = indoor_dry_bulb_temperature_norm_noisy * (high_limit - low_limit) + low_limit

    #     # update temperature
    #     self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = indoor_dry_bulb_temperature

    # def get_dynamics_input(self) -> np.ndarray:
    #     model_input = []
    #     for i, k in enumerate(self.dynamics.input_observation_names):
    #         if k == 'indoor_dry_bulb_temperature':
    #             model_input.append(self.dynamics._model_input[i][:-1])
    #         else:
    #             model_input.append(self.dynamics._model_input[i][1:])
    #     model_input = np.array(model_input, dtype='float32')
    #     return model_input

    # def _update_dynamics_input(self):
    #     """Updates and returns the input time series for the dynamics prediction model."""
    #     observations = self.observations(include_all=True, normalize=False, periodic_normalization=True)
        
    #     # Check if we have None values that need to be filled
    #     has_none_values = any(None in l for l in self.dynamics._model_input)
        
    #     if has_none_values:
    #         # For the initial steps, fill None values with the current observation
    #         for i, (l, k, min_, max_) in enumerate(zip(
    #             self.dynamics._model_input,
    #             self.dynamics.input_observation_names,
    #             self.dynamics.input_normalization_minimum,
    #             self.dynamics.input_normalization_maximum
    #         )):
    #             normalized_obs = (observations[k] - min_) / (max_ - min_)
    #             # Replace None values with current observation and add new observation
    #             updated_list = [normalized_obs if val is None else val for val in l]
    #             # Take the last lookback elements and add the new observation
    #             self.dynamics._model_input[i] = updated_list[-self.dynamics.lookback:] + [normalized_obs]
    #     else:
    #         # Normal operation: slide the window and add new observation
    #         self.dynamics._model_input = [
    #             l[-self.dynamics.lookback:] + [(observations[k] - min_) / (max_ - min_)]
    #             for l, k, min_, max_ in zip(
    #                 self.dynamics._model_input,
    #                 self.dynamics.input_observation_names,
    #                 self.dynamics.input_normalization_minimum,
    #                 self.dynamics.input_normalization_maximum
    #             )
    #         ]

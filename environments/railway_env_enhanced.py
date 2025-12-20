import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List
from enum import Enum

class SignalAspect(Enum):
    """Railway signal aspects (ERTMS/ETCS)"""
    RED = 0        # Stop
    YELLOW = 1     # Proceed with caution
    DOUBLE_YELLOW = 2  # Prepare to reduce speed
    GREEN = 3      # Proceed

class TrackCircuitState(Enum):
    """Track circuit occupancy states"""
    OCCUPIED = 0
    CLEAR = 1
    BROKEN = 2  # Fault state

class EnhancedRailwayEnv(gym.Env):
    """
    Enhanced Train Movement Authority Environment for Formal Verification
    Based on ERTMS/ETCS Level 2 specifications
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # ========== REALISTIC RAILWAY STATE DIMENSIONS ==========
        # Total: 28 dimensions (expandable)
        self.state_dim = 28
        self.action_dim = 4  # Same action space
        
        # State indices
        self.STATE_INDICES = {
            # Core train states (0-5)
            'position': 0,           # meters from start
            'speed': 1,              # km/h
            'acceleration': 2,       # m/s²
            'braking_distance': 3,   # meters
            'train_length': 4,       # meters
            'train_mass': 5,         # tons
            
            # Track & infrastructure (6-13)
            'gradient': 6,           # % incline
            'curvature': 7,          # radius in meters
            'track_quality': 8,      # 0-1 (poor-excellent)
            'next_signal': 9,        # SignalAspect enum value
            'signal_distance': 10,   # meters to next signal
            'track_circuit': 11,     # TrackCircuitState
            'section_occupancy': 12, # 0-1 (occupied ratio)
            'switch_position': 13,   # 0-1 (0=normal, 1=reverse)
            
            # Primary safety sensors (14-21) - ERTMS Level 2
            'axle_counter': 14,      # 0-1 (clear/occupied)
            'balise_telegram': 15,   # Last received balise data
            'radio_block_center': 16, # RBC communication status
            'train_integrity': 17,    # 0-1 (train complete)
            'door_status': 18,        # 0=closed, 1=open
            'brake_pressure': 19,     # bar
            'wheel_slip': 20,         # 0-1 probability
            'sand_system': 21,        # 0-1 (available)
            
            # Environmental sensors (22-27)
            'weather_condition': 22,  # 0-1 (clear-storm)
            'visibility': 23,         # meters
            'temperature': 24,        # °C
            'wind_speed': 25,         # m/s
            'rain_intensity': 26,     # mm/h
            'adhesion_level': 27      # 0-1 (poor-excellent)
        }
        
        # ========== REALISTIC PARAMETERS ==========
        # Based on ERTMS/ETCS specifications
        self.PARAMS = {
            'max_speed': config.get('max_speed', 160),  # km/h
            'max_acceleration': 0.5,  # m/s²
            'emergency_deceleration': -2.73,  # m/s² (ERTMS requirement)
            'service_deceleration': -0.7,     # m/s²
            'train_length_range': [100, 400],  # meters
            'train_mass_range': [200, 1000],   # tons
            'gradient_range': [-4, 4],         # % (uphill positive)
            'curvature_range': [300, 2000],    # meters radius
            'braking_curve_margin': 1.15,      # Safety margin
            'movement_authority_distance': 5000,  # meters
        }
        
        # ========== SAFETY PARAMETERS ==========
        self.safety_margins = {
            'sighting_distance': 200,  # Distance to see obstruction
            'reaction_time': 2.5,      # Driver reaction time (seconds)
            'braking_response': 3.0,   # Brake system response time
            'position_uncertainty': 5, # GPS/balise position uncertainty (m)
            'speed_uncertainty': 0.1,  # Speed measurement uncertainty
        }
        
        # ========== INITIALIZATION ==========
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.violations = {'adequacy': 0, 'consistency': 0, 
                          'stability': 0, 'timeliness': 0}
        
        # For formal verification
        self.trace_history = []  # Store state-action pairs for verification
        self.ctl_properties_satisfied = []  # Track CTL property satisfaction
        
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset to initial state with realistic railway parameters"""
        super().reset(seed=seed)
        
        s = self.state
        idx = self.STATE_INDICES
        
        # 1. Train parameters (realistic ranges)
        s[idx['position']] = 0.0
        s[idx['speed']] = np.random.uniform(0, 80)  # Initial speed 0-80 km/h
        s[idx['acceleration']] = 0.0
        s[idx['train_length']] = np.random.uniform(*self.PARAMS['train_length_range'])
        s[idx['train_mass']] = np.random.uniform(*self.PARAMS['train_mass_range'])
        
        # Calculate braking distance based on current speed and mass
        s[idx['braking_distance']] = self._calculate_braking_distance(
            s[idx['speed']], s[idx['train_mass']])
        
        # 2. Track infrastructure
        s[idx['gradient']] = np.random.uniform(*self.PARAMS['gradient_range'])
        s[idx['curvature']] = np.random.uniform(*self.PARAMS['curvature_range'])
        s[idx['track_quality']] = np.random.uniform(0.7, 1.0)
        s[idx['next_signal']] = SignalAspect.GREEN.value
        s[idx['signal_distance']] = np.random.uniform(500, 2000)
        s[idx['track_circuit']] = TrackCircuitState.CLEAR.value
        s[idx['section_occupancy']] = 0.0
        s[idx['switch_position']] = 0.0  # Normal position
        
        # 3. Primary safety sensors (with realistic noise)
        s[idx['axle_counter']] = 1.0  # Clear
        s[idx['balise_telegram']] = self._generate_balise_telegram()
        s[idx['radio_block_center']] = 1.0  # Connected
        s[idx['train_integrity']] = 1.0  # Train complete
        s[idx['door_status']] = 0.0  # Closed
        s[idx['brake_pressure']] = np.random.uniform(3.8, 4.2)  # Normal range
        s[idx['wheel_slip']] = 0.0
        s[idx['sand_system']] = 1.0  # Available
        
        # 4. Environmental conditions
        s[idx['weather_condition']] = np.random.uniform(0, 0.3)  # Mostly clear
        s[idx['visibility']] = np.random.uniform(500, 2000)  # meters
        s[idx['temperature']] = np.random.uniform(5, 35)  # °C
        s[idx['wind_speed']] = np.random.uniform(0, 15)  # m/s
        s[idx['rain_intensity']] = np.random.uniform(0, 5)  # mm/h
        
        # Calculate adhesion based on conditions
        s[idx['adhesion_level']] = self._calculate_adhesion(
            s[idx['weather_condition']], 
            s[idx['temperature']],
            s[idx['rain_intensity']])
        
        # Initialize timing
        self.decision_time = 0
        self.last_ma_grant_time = 0
        self.ma_valid_until = 0
        
        # Clear history for new episode
        self.trace_history = []
        self.ctl_properties_satisfied = []
        
        return s.copy(), {'adhesion': s[idx['adhesion_level']]}
    
    def _calculate_braking_distance(self, speed_kmh: float, mass_tons: float) -> float:
        """Calculate braking distance based on ERTMS braking curves"""
        speed_ms = speed_kmh / 3.6  # Convert to m/s
        
        # Emergency braking distance (ERTMS formula simplified)
        deceleration = abs(self.PARAMS['emergency_deceleration'])
        
        # Adjust for mass (heavier trains brake slower)
        mass_factor = 1.0 + (mass_tons - 400) / 1000  # Small adjustment
        
        # Adjust for gradient
        gradient = self.state[self.STATE_INDICES['gradient']]
        gradient_factor = 1.0 + gradient / 100  # Uphill helps braking
        
        distance = (speed_ms ** 2) / (2 * deceleration * mass_factor * gradient_factor)
        
        # Add safety margins
        distance *= self.PARAMS['braking_curve_margin']
        distance += self.safety_margins['position_uncertainty']
        
        return distance
    
    def _calculate_adhesion(self, weather: float, temp: float, rain: float) -> float:
        """Calculate wheel-rail adhesion coefficient (0-1)"""
        # Base adhesion on dry rail
        adhesion = 0.3  # Dry rail (typical)
        
        # Reduce for rain
        if rain > 0.1:
            adhesion -= 0.1 * min(rain / 10, 1.0)
        
        # Reduce for extreme weather
        if weather > 0.7:  # Storm conditions
            adhesion -= 0.15
        
        # Temperature effects (ice risk below 0°C)
        if temp < 2.0:
            adhesion -= 0.05 * (2.0 - temp) / 2.0
        
        return max(0.05, adhesion)  # Minimum adhesion
    
    def _generate_balise_telegram(self) -> float:
        """Simulate Eurobalise telegram data"""
        telegram_info = {
            'absolute_position': np.random.uniform(0, 10000),
            'next_signal_distance': np.random.uniform(500, 2000),
            'permitted_speed': np.random.uniform(80, self.PARAMS['max_speed']),
            'gradient_ahead': np.random.uniform(-3, 3),
        }
        return hash(str(telegram_info)) % 1000 / 1000.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with realistic railway dynamics"""
        
        # Store for formal verification trace
        self.trace_history.append({
            'state': self.state.copy(),
            'action': action,
            'time': self.decision_time
        })
        
        # Calculate reward (WITH FIXED SCALING)
        reward = self._calculate_enhanced_reward(action)
        
        # Update train dynamics
        self._update_train_dynamics(action)
        
        # Update environment and sensors
        self._update_environment()
        self._update_sensors()
        
        # Check credibility violations
        violations = self._check_enhanced_credibility_violations(action)
        self.violations = {k: self.violations[k] + v for k, v in violations.items()}
        
        # Penalty for violations (REDUCED FROM 50.0 to 2.0)
        violation_penalty = sum(violations.values()) * 2.0
        reward -= violation_penalty
        
        # Check termination conditions
        terminated = self._check_enhanced_termination()
        truncated = self.decision_time > 1000  # Max decision steps
        
        # Additional info for verification
        info = {
            'violations': violations,
            'total_violations': self.violations,
            'action': action,
            'position': self.state[self.STATE_INDICES['position']],
            'speed': self.state[self.STATE_INDICES['speed']],
            'braking_distance': self.state[self.STATE_INDICES['braking_distance']],
            'safety_margin': self._calculate_safety_margin(),
            'ctl_properties': self._check_ctl_properties(),
            'trace': self.trace_history[-10:] if self.trace_history else []
        }
        
        # Update timing
        self.decision_time += 1
        
        # Update Movement Authority validity
        if action == 0:  # Grant_MA
            self.last_ma_grant_time = self.decision_time
            self.ma_valid_until = self.decision_time + 30  # 30 steps validity
        
        # CLIP FINAL REWARD to prevent extremes
        reward = np.clip(reward, -10.0, 10.0)
        
        return self.state.copy(), float(reward), terminated, truncated, info
    
    def _calculate_enhanced_reward(self, action: int) -> float:
        """Enhanced reward function with SCALED penalties"""
        idx = self.STATE_INDICES
        s = self.state
        
        base_reward = 0.0
        
        # 1. Efficiency rewards (SCALED DOWN)
        if action == 0:  # Grant_MA
            optimal_speed = self.PARAMS['max_speed'] * 0.8  # 80% of max
            speed_ratio = min(s[idx['speed']] / optimal_speed, 1.0)
            base_reward += 0.5 * speed_ratio  # REDUCED from 2.0
            base_reward += 0.05 * (s[idx['speed']] / self.PARAMS['max_speed'])  # Reduced
        
        elif action == 3:  # Reduce_Speed (safe but inefficient)
            base_reward += 0.2  # REDUCED from 0.5
        
        # 2. Safety rewards/penalties (MASSIVELY REDUCED)
        safety_status = self._check_safety_status()
        if not safety_status['safe_to_proceed']:
            base_reward -= 1.0  # REDUCED from 5.0 (MASSIVE REDUCTION)
        else:
            base_reward += 0.1  # Small reward for safe state
        
        # 3. Energy efficiency (very small penalty)
        if s[idx['acceleration']] > 0:
            base_reward -= 0.01 * s[idx['acceleration']]  # Reduced
        
        # 4. Passenger comfort (smooth operation)
        jerk = abs(s[idx['acceleration']] - self.prev_acceleration if hasattr(self, 'prev_acceleration') else 0)
        base_reward -= 0.01 * jerk  # Reduced
        
        self.prev_acceleration = s[idx['acceleration']]
        
        # 5. Add progress reward
        progress = s[idx['position']] / 10000.0  # Normalized progress (0-1)
        base_reward += 0.1 * progress
        
        return base_reward
    
    def _check_safety_status(self) -> Dict:
        """Comprehensive safety check based on ERTMS rules"""
        idx = self.STATE_INDICES
        s = self.state
        
        status = {
            'safe_to_proceed': True,
            'reasons': [],
            'safety_margin': self._calculate_safety_margin()
        }
        
        braking_distance = s[idx['braking_distance']]
        distance_to_obstacle = self._distance_to_next_obstacle()
        
        if braking_distance > distance_to_obstacle:
            status['safe_to_proceed'] = False
            status['reasons'].append(f"Braking distance ({braking_distance:.1f}m) > Obstacle distance ({distance_to_obstacle:.1f}m)")
        
        if s[idx['next_signal']] == SignalAspect.RED.value and s[idx['speed']] > 0:
            status['safe_to_proceed'] = False
            status['reasons'].append("Passing red signal")
        
        max_permitted_speed = self._get_max_permitted_speed()
        if s[idx['speed']] > max_permitted_speed * 1.05:
            status['safe_to_proceed'] = False
            status['reasons'].append(f"Speed ({s[idx['speed']]:.1f}km/h) > Permitted ({max_permitted_speed:.1f}km/h)")
        
        if s[idx['train_integrity']] < 0.9:
            status['safe_to_proceed'] = False
            status['reasons'].append("Train integrity compromised")
        
        if s[idx['door_status']] > 0.1:
            status['safe_to_proceed'] = False
            status['reasons'].append("Doors not fully closed")
        
        return status
    
    def _calculate_safety_margin(self) -> float:
        """Calculate safety margin ratio (0-1, higher is safer)"""
        idx = self.STATE_INDICES
        s = self.state
        
        braking_distance = s[idx['braking_distance']]
        obstacle_distance = self._distance_to_next_obstacle()
        
        if obstacle_distance <= 0:
            return 0.0
        
        margin = (obstacle_distance - braking_distance) / braking_distance
        return max(0.0, min(1.0, margin))  # Clip to 0-1
    
    def _distance_to_next_obstacle(self) -> float:
        """Calculate distance to next obstacle (simplified)"""
        idx = self.STATE_INDICES
        s = self.state
        
        distance = s[idx['signal_distance']]
        if s[idx['track_circuit']] == TrackCircuitState.OCCUPIED.value:
            distance *= 0.3
        if np.random.random() < 0.01:
            distance = np.random.uniform(50, 500)
        return distance
    
    def _get_max_permitted_speed(self) -> float:
        """Get maximum permitted speed based on track conditions"""
        idx = self.STATE_INDICES
        s = self.state
        
        base_speed = self.PARAMS['max_speed']
        
        if s[idx['curvature']] < 800:
            base_speed = min(base_speed, 80)
        elif s[idx['curvature']] < 1200:
            base_speed = min(base_speed, 120)
        
        if s[idx['adhesion_level']] < 0.2:
            base_speed *= 0.7
        elif s[idx['adhesion_level']] < 0.3:
            base_speed *= 0.8
        
        if s[idx['visibility']] < 300:
            base_speed = min(base_speed, 60)
        
        return base_speed
    
    def _update_train_dynamics(self, action: int):
        """Realistic train dynamics update"""
        idx = self.STATE_INDICES
        s = self.state
        
        dt = 1.0  # 1 second time step
        
        position = s[idx['position']]
        speed_kmh = s[idx['speed']]
        speed_ms = speed_kmh / 3.6
        acceleration = s[idx['acceleration']]
        gradient = s[idx['gradient']] / 100  # Convert % to ratio
        train_mass = s[idx['train_mass']]
        adhesion = s[idx['adhesion_level']]
        
        if action == 0:  # Grant_MA - normal operation
            target_acceleration = min(self.PARAMS['max_acceleration'], 0.3)
        elif action == 1:  # Deny_MA - coast
            target_acceleration = -0.1
        elif action == 3:  # Reduce_Speed - brake gently
            target_acceleration = self.PARAMS['service_deceleration'] * 0.5
        else:  # Request_Update - maintain current
            target_acceleration = 0.0
        
        gradient_effect = -9.81 * np.sin(np.arctan(gradient))
        target_acceleration += gradient_effect
        
        max_acceleration = adhesion * 0.5  # Simplified
        target_acceleration = np.clip(target_acceleration, 
                                     -abs(self.PARAMS['emergency_deceleration']),
                                     max_acceleration)
        
        acceleration_alpha = 0.3  # Smoothing factor
        s[idx['acceleration']] = (acceleration_alpha * target_acceleration + 
                                 (1 - acceleration_alpha) * acceleration)
        
        new_speed_ms = max(0, speed_ms + s[idx['acceleration']] * dt)
        s[idx['speed']] = new_speed_ms * 3.6  # Back to km/h
        
        s[idx['position']] = position + new_speed_ms * dt
        
        s[idx['braking_distance']] = self._calculate_braking_distance(
            s[idx['speed']], train_mass)
    
    def _update_environment(self):
        """Update environmental conditions"""
        idx = self.STATE_INDICES
        s = self.state
        
        s[idx['weather_condition']] = np.clip(
            s[idx['weather_condition']] + np.random.normal(0, 0.01),
            0, 1)
        
        s[idx['temperature']] += np.random.normal(0, 0.1)
        s[idx['rain_intensity']] = np.clip(
            s[idx['rain_intensity']] + np.random.normal(0, 0.05),
            0, 20)
        
        s[idx['adhesion_level']] = self._calculate_adhesion(
            s[idx['weather_condition']],
            s[idx['temperature']],
            s[idx['rain_intensity']])
        
        s[idx['signal_distance']] = max(0, s[idx['signal_distance']] - 
                                       (s[idx['speed']] / 3.6))  # Convert to m/s
        
        if np.random.random() < 0.02:  # 2% chance per step
            s[idx['next_signal']] = np.random.choice([
                SignalAspect.GREEN.value,
                SignalAspect.YELLOW.value,
                SignalAspect.RED.value
            ], p=[0.7, 0.2, 0.1])
    
    def _update_sensors(self):
        """Update sensor readings with realistic noise and failures"""
        idx = self.STATE_INDICES
        s = self.state
        
        sensor_config = self.config.get('sensors', {})
        failure_rate = sensor_config.get('failure_rate', 0.001)
        noise_level = sensor_config.get('noise_level', 0.02)
        
        primary_sensors = [
            idx['axle_counter'],
            idx['brake_pressure'],
            idx['wheel_slip'],
            idx['train_integrity']
        ]
        
        for sensor_idx in primary_sensors:
            if np.random.random() < failure_rate:
                s[sensor_idx] = np.nan
            else:
                noise = np.random.normal(0, noise_level * (s[sensor_idx] if s[sensor_idx]!=0 else 1.0))
                s[sensor_idx] += noise
                
                if sensor_idx == idx['brake_pressure']:
                    s[sensor_idx] = np.clip(s[sensor_idx], 0, 5)
                elif sensor_idx in [idx['axle_counter'], idx['train_integrity'], idx['wheel_slip']]:
                    s[sensor_idx] = np.clip(s[sensor_idx], 0, 1)
    
    def _check_enhanced_credibility_violations(self, action: int) -> Dict[str, int]:
        """Enhanced credibility checks for formal verification"""
        violations = {'adequacy': 0, 'consistency': 0, 
                     'stability': 0, 'timeliness': 0}
        
        critical_sensors = [
            self.STATE_INDICES['speed'],
            self.STATE_INDICES['position'],
            self.STATE_INDICES['braking_distance'],
            self.STATE_INDICES['train_integrity']
        ]
        
        for sensor_idx in critical_sensors:
            val = self.state[sensor_idx]
            if np.isnan(val) or np.isinf(val):
                violations['adequacy'] = 1
                break
        
        speed_odometer = self.state[self.STATE_INDICES['speed']]
        speed_gps = speed_odometer * (1 + np.random.normal(0, 0.01))
        if abs(speed_odometer - speed_gps) / max(speed_odometer, 1) > 0.05:
            violations['consistency'] = 1
        
        if hasattr(self, 'prev_speed'):
            speed_change = abs(self.state[self.STATE_INDICES['speed']] - self.prev_speed)
            if speed_change > 20:  # More than 20 km/h change in one step
                violations['stability'] = 1
        self.prev_speed = self.state[self.STATE_INDICES['speed']]
        
        if action == 0:  # Grant_MA
            if self.decision_time > self.ma_valid_until:
                violations['timeliness'] = 1
        
        return violations
    
    def _check_enhanced_termination(self) -> bool:
        """Enhanced termination conditions"""
        idx = self.STATE_INDICES
        s = self.state
        
        if s[idx['position']] >= 10000:  # 10km route
            return True
        
        safety = self._check_safety_status()
        if not safety['safe_to_proceed'] and 'Passing red signal' in safety['reasons']:
            return True
        
        if s[idx['speed']] < 1 and s[idx['position']] > 100:
            return True
        
        return False
    
    def _check_ctl_properties(self) -> List[str]:
        """Check CTL properties for formal verification"""
        idx = self.STATE_INDICES
        s = self.state
        
        satisfied = []
        
        if s[idx['speed']] > 0:
            braking_distance = s[idx['braking_distance']]
            obstacle_distance = self._distance_to_next_obstacle()
            if braking_distance < obstacle_distance:
                satisfied.append("AG(speed>0 → braking<obstacle)")
        
        if not (s[idx['next_signal']] == SignalAspect.RED.value and s[idx['speed']] > 5):
            satisfied.append("AG¬(red_signal ∧ high_speed)")
        
        if s[idx['door_status']] < 0.1 or s[idx['speed']] < 1:
            satisfied.append("AG(doors_closed ∨ stopped)")
        
        return satisfied
    
    def get_formal_model(self) -> Dict:
        """Export environment as formal model for verification"""
        idx = self.STATE_INDICES
        
        return {
            'states': {
                'position': float(self.state[idx['position']]),
                'speed': float(self.state[idx['speed']]),
                'braking_distance': float(self.state[idx['braking_distance']]),
                'signal_aspect': int(self.state[idx['next_signal']]),
                'signal_distance': float(self.state[idx['signal_distance']]),
                'train_integrity': float(self.state[idx['train_integrity']]),
                'door_status': float(self.state[idx['door_status']]),
                'track_circuit': int(self.state[idx['track_circuit']])
            },
            'actions': ['Grant_MA', 'Deny_MA', 'Request_Update', 'Reduce_Speed'],
            'safety_properties': self._check_ctl_properties(),
            'violations': self.violations,
            'trace_length': len(self.trace_history)
        }
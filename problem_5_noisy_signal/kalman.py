from SignalReceiver import SignalReceiver

# from https://github.com/lblackhall/pyconau2016
class SingleStateKalmanFilter(object):

    def __init__(self, A, B, C, x, P, Q, R):
        self.A = A  # Process dynamics
        self.B = B  # Control dynamics
        self.C = C  # Measurement dynamics
        self.current_state_estimate = x  # Current state estimate
        self.current_prob_estimate = P  # Current probability of state estimate
        self.Q = Q  # Process covariance
        self.R = R  # Measurement covariance

    def current_state(self):
        return self.current_state_estimate

    def step(self, control_input, measurement):
        # Prediction step
        predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_input
        predicted_prob_estimate = (self.A * self.current_prob_estimate) * self.A + self.Q

        # Observation step
        innovation = measurement - self.C * predicted_state_estimate
        innovation_covariance = self.C * predicted_prob_estimate * self.C + self.R

        # Update step
        kalman_gain = predicted_prob_estimate * self.C * 1 / float(innovation_covariance)
        self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation

        # eye(n) = nxn identity matrix.
        self.current_prob_estimate = (1 - kalman_gain * self.C) * predicted_prob_estimate

def baseline(signal_nr):
    sr = SignalReceiver(signal_nr)
    val = sr.get_value()
    while val:
        sr.push_value(val)
        val = sr.get_value()
    print("Signal {0} baseline error: {1}".format(signal_nr, sr.get_error()))
    print("-"*50)

def with_kalman_filter(signal_nr):
    sr = SignalReceiver(signal_nr)
    val = sr.get_value()

    A = 0.494
    B = 1
    C = 1
    Q = 0.05
    R = 0.001
    x = val
    P = 1

    filtr = SingleStateKalmanFilter(A,B,C,x,P,Q,R)

    while val:
        filtr.step(0,val)
        sr.push_value(filtr.current_state())
        val = sr.get_value()
    print("\nSignal {0} after kalman filtering error: {1}".format(signal_nr, sr.get_error()))

kinda_success = [3,4,5]

for success_nr in kinda_success:
    with_kalman_filter(success_nr)
    baseline(success_nr)

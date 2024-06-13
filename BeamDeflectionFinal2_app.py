import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate moment of inertia
def calcMomentOfInertia(shape, dimensions, E):
    if shape == 'Circle':
        d = dimensions[0]
        I = (np.pi * d**4) / 64
    elif shape == 'Square':
        a = dimensions[0]
        I = (a**4) / 12
    elif shape == 'Rectangle':
        b, h = dimensions
        I = (b * h**3) / 12
    elif shape == 'I-Beam':
        b_f, h_f, b_w, h_w = dimensions
        I_f = (b_f * h_f**3) / 12
        I_w = (b_w * h_w**3) / 12
        I = 2 * I_f + I_w  # Simplified for demonstration
    
    I *= E  # Multiply by Young's modulus
    
    return I

# Function to calculate deflection and rotation for point load
def calcDeflectionPointLoad(P, L, a, EI, delX):
    X = np.arange(0, L + delX, delX)
    M = np.zeros_like(X)
    for i, x in enumerate(X):
        if x < a:
            M[i] = P * x
        else:
            M[i] = P * a

    return calcDeflection(M, EI, delX, 0, 0, 0), X

# Function to calculate deflection and rotation for uniformly distributed load
def calcDeflectionUDL(w, L, EI, delX):
    X = np.arange(0, L + delX, delX)
    M = (w * X**2) / 2

    return calcDeflection(M, EI, delX, 0, 0, 0), X

# Function to calculate deflection and rotation
def calcDeflection(M, EI, delX, theta_0, v_0, supportIndexA):
    theta_im1 = theta_0
    v_im1 = v_0

    Rotation = np.zeros(len(M))
    Rotation[supportIndexA] = theta_im1
    Deflection = np.zeros(len(M))
    Deflection[supportIndexA] = v_im1

    for i, m in enumerate(M[supportIndexA:]):
        ind = i + supportIndexA
        if i > 0:
            M_im1 = M[ind - 1]
            M_i = M[ind]
            M_avg = 0.5 * (M_i + M_im1)

            theta_i = theta_im1 + (M_avg / EI) * delX
            v_i = v_im1 + 0.5 * (theta_i + theta_im1) * delX

            Rotation[ind] = theta_i
            Deflection[ind] = v_i

            theta_im1 = theta_i
            v_im1 = v_i

    return Rotation, Deflection

# Streamlit UI
st.title("Beam Deflection Calculator")

st.write("""
This application calculates the deflection and rotation of a beam based on its cross-sectional shape, Young's modulus (E),
load type, and other parameters. Please select the appropriate options and enter the required values to see the results.
""")

shape = st.selectbox("Select the cross-sectional shape of the beam:", ["Circle", "Square", "Rectangle", "I-Beam"])

if shape == 'Circle':
    d = st.number_input("Enter the diameter (d) in meters:", value=0.1)
    dimensions = [d]
elif shape == 'Square':
    a = st.number_input("Enter the side length (a) in meters:", value=0.1)
    dimensions = [a]
elif shape == 'Rectangle':
    b = st.number_input("Enter the base width (b) in meters:", value=0.1)
    h = st.number_input("Enter the height (h) in meters:", value=0.2)
    dimensions = [b, h]
elif shape == 'I-Beam':
    b_f = st.number_input("Enter the flange width (b_f) in meters:", value=0.1)
    h_f = st.number_input("Enter the flange height (h_f) in meters:", value=0.02)
    b_w = st.number_input("Enter the web width (b_w) in meters:", value=0.02)
    h_w = st.number_input("Enter the web height (h_w) in meters:", value=0.1)
    dimensions = [b_f, h_f, b_w, h_w]

E = st.number_input("Enter the Young's modulus (E) in Pa:", value=21000000.0)

I = calcMomentOfInertia(shape, dimensions, E)

# Display the calculated moment of inertia (I)
st.write(f"### Calculated Moment of Inertia (I): {I:.2f} m^4")

L = st.number_input("Enter the length of the beam (L) in meters:", value=10.0)
delX = st.number_input("Enter the step size (delX) in meters:", value=0.1)
load_type = st.selectbox("Select the load type:", ["Point Load", "Uniformly Distributed Load"])

P = None
a = None
w = None

if load_type == "Point Load":
    P = st.number_input("Enter the magnitude of the point load (P) in Newtons:", value=1000.0)
    a = st.number_input("Enter the distance from the left support to the point load (a) in meters:", value=5.0)
else:
    w = st.number_input("Enter the magnitude of the uniformly distributed load (w) in N/m:", value=500.0)

if st.button("Calculate"):
    if load_type == "Point Load":
        (Rotation, Deflection), X = calcDeflectionPointLoad(P, L, a, E * I, delX)
    else:
        (Rotation, Deflection), X = calcDeflectionUDL(w, L, E * I, delX)

    # Create a dataframe to display the results
    results = pd.DataFrame({
        'Position (m)': X,
        'Rotation (radians)': Rotation,
        'Deflection (m)': Deflection
    })

    st.write("### Results:")
    st.dataframe(results)

    # Plot the results
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Deflection (m)', color=color)
    ax1.plot(X, Deflection, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Rotation (radians)', color=color)
    ax2.plot(X, Rotation, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    st.pyplot(fig)

    st.write("""
    #### Note:
    - The position, rotation, and deflection values are calculated and displayed in the table above.
    - The plot shows the rotation (blue line) and deflection (red line) of the beam along its length.
    """)

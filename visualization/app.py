"""Streamlit app to visualize runs results.
"""
import streamlit as st
import json
import os
from streamlit_agraph import agraph, Node, Edge, Config
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
from PIL.Image import Image
from typing import Dict, Union
import base64


def load_data(file_path: str) -> Dict[str, Union[str, float]]:
    """Load dictionary with molecules from a specific .json file
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    # In situ modification for reaction data
    for key in data.keys():
        for key, value in data[key]["synthesis_data"].items():
            if value["is_rxn"] == True:
                rxn = value["rxn_smiles"]
                reactants = rxn.split(">>")[0].split(".")
                products = rxn.split(">>")[1].split(".")
                value["children"] = reactants
                value["parent"] = products

    return data


def smiles_to_image(smiles: str) -> Image:
    """Load PIL image from SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol)


def smiles_to_base64_url(smiles: str) -> str:
    """Convert SMILES to base64 image as a string.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(300, 300), dpi=800)  

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"


def create_synthesis_graph(molecule_data: Dict[str, Union[str, float]]
                           ) -> Union[Node, Edge]:
    """Function to parse nodes and edges into final structure of nodes and edges
    """
    nodes = []
    edges = []

    synthesis_data = molecule_data["synthesis_data"]
    enforced_block = molecule_data.get("enforced_block", None)

    # Generate nodes
    for key, value in synthesis_data.items():

        if value['is_mol']:
            
            if value["mol_smiles"] == enforced_block:
                color = "blue"

            elif value["is_purchasable"]:
                color = "green"
            
            else:
                color = "red"
            
            nodes.append(Node(id=value["mol_smiles"], 
                              label='', 
                              size=60, 
                              shape="circularImage",
                              color=color,
                              image=smiles_to_base64_url(value['mol_smiles'])))
        else:
            # Create reaction node
            rxn_label = f"{value['rxn_name']} ({value['rxn_class']})" \
                if value['rxn_name'] else "Unknown Reaction"
            nodes.append(Node(id=value["rxn_smiles"], 
                              label=rxn_label, 
                              size=10, 
                              color="blue",
                              shape="diamond"))
        
    # Create edges
    for key, value in synthesis_data.items():
        if value["is_rxn"]:
            id = value["rxn_smiles"]
            parent_id = value["parent"][0]
            edges.append(Edge(source=id, target=parent_id))

            for child in value["children"]:
                edges.append(Edge(source=child, target=id))

    return nodes, edges

# Load and parse data
top_graphs_folder = st.secrets["file_path"]
experiment_files = [f for f in os.listdir(top_graphs_folder) if f.endswith("top_graphs.json")]

# Sidebar for selecting experiment
st.sidebar.header("Select experiment")
selected_experiment = st.sidebar.selectbox("Experiments", experiment_files)

# Display the results associated to the specified experiment
if selected_experiment:

    file_path = os.path.join(top_graphs_folder, selected_experiment)
    data = load_data(file_path)

    # Get molecules sorted by total reward
    gen_molecules = sorted(data.keys(), 
                           key=lambda x: data[x]["reward"]["reward"], 
                           reverse=True)

    # Streamlit app
    st.title("Molecule Synthesis Visualization")

    # Sidebar for SMILES input and route selection
    st.sidebar.header("Generate molecules")
    mol = st.sidebar.selectbox("Select molecule", options=gen_molecules)

    if mol:
        molecule_data = data[mol]

        # Display reward values
        st.sidebar.subheader("Reward Values")
        st.sidebar.write(molecule_data["reward"])

        nodes, edges = create_synthesis_graph(molecule_data)

        config = Config(width=1000, height=1000, directed=True, physics=False, hierarchical=True)
        agraph(nodes=nodes, edges=edges, config=config)
        # Display image of selected molecule
        st.image(smiles_to_image(mol), caption="Generated molecule", width=200)
            
    else:
        st.sidebar.write("Error fetching molecules from input file")

else: 
    st.sidebar.write("Error loading experiment")
import streamlit as st
import json
from streamlit_agraph import agraph, Node, Edge, Config
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

# Load JSON data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # in situ modification for reaction data
    for key in data.keys():
        for key, value in data[key]["synthesis_data"].items():
            if value["is_rxn"] == True:
                rxn = value["rxn_smiles"]
                reactants = rxn.split(">>")[0].split(".")
                products = rxn.split(">>")[1].split(".")
                value["children"] = reactants
                value["parent"] = products

    return data


# Function to create RDKit image from SMILES
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol)

# Convert SMILES to base64 image data URL
def smiles_to_base64_url(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(300, 300), dpi=800)  # Adjust size as needed
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


# Display synthesis path using nodes and edges
def create_synthesis_graph(molecule_data):
    """Function to parse nodes and edges into final structure"""
    nodes = []
    edges = []

    synthesis_data = molecule_data["synthesis_data"]
    enforced_block = molecule_data["enforced_block"]
    # generate nodes
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
        
    # create edges
    for key, value in synthesis_data.items():
        if value["is_rxn"]:
            id = value["rxn_smiles"]
            parent_id = value["parent"][0]
            edges.append(Edge(source=id, target=parent_id))

            for child in value["children"]:
                edges.append(Edge(source=child, target=id))

    return nodes, edges

# Load and parse data
data_file = st.secrets["file_path"]  # Use your uploaded JSON file path
data = load_data(data_file)

gen_molecules = list(data.keys())

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

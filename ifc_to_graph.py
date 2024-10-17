import ifcopenshell
import ifcopenshell.util.element

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from typing import List

def convert_ifc_to_graph_document(ifc_file_path) -> List[GraphDocument]:
    model = ifcopenshell.open(ifc_file_path)

    nodes = []
    ifc_objects={}
    for ifc_object in model.by_type("IfcObject"):
        if ifc_object.is_a("IfcVirtualElement"):
            continue

        type = ifcopenshell.util.element.get_type(ifc_object)
        if type is not None:
            psets = ifcopenshell.util.element.get_psets(type)

        ifc_objects[ifc_object.id()] = {
            "id": f'#{str(ifc_object.id())}',
            "type":ifc_object.is_a()
        }

        nodes.append(
                Node(
                    id=ifc_objects[ifc_object.id()]["id"],
                    type=ifc_objects[ifc_object.id()]["type"],
                    properties={
                        "type":ifc_objects[ifc_object.id()]["type"],
                        "name": ifc_object.Name
                    },
                )
            )
        
    relationships = []
    for ifc_rel in model.by_type("IfcRelationship"):

        relating_object = None
        related_objects = []

        if ifc_rel.is_a("IfcRelAggregates"):
            relating_object = ifc_rel.RelatingObject
            related_objects = ifc_rel.RelatedObjects
        elif ifc_rel.is_a("IfcRelNests"):
            relating_object = ifc_rel.RelatingObject
            related_objects = ifc_rel.RelatedObjects
        elif ifc_rel.is_a("IfcRelAssignsToGroup"):
            relating_object = ifc_rel.RelatingGroup
            related_objects = ifc_rel.RelatedObjects
        elif ifc_rel.is_a("IfcRelConnectsElements"):
            relating_object = ifc_rel.RelatingElement
            related_objects = [ifc_rel.RelatedElement]
        elif ifc_rel.is_a("IfcRelConnectsStructuralMember"):
            relating_object = ifc_rel.RelatingStructuralMember
            related_objects = [ifc_rel.RelatedStructuralConnection]
        elif ifc_rel.is_a("IfcRelContainedInSpatialStructure"):
            relating_object = ifc_rel.RelatingStructure
            related_objects = ifc_rel.RelatedElements
        elif ifc_rel.is_a("IfcRelFillsElement"):
            relating_object = ifc_rel.RelatingOpeningElement
            related_objects = [ifc_rel.RelatedBuildingElement]
        elif ifc_rel.is_a("IfcRelVoidsElement"):
            relating_object = ifc_rel.RelatingBuildingElement
            related_objects = [ifc_rel.RelatedOpeningElement]
        elif ifc_rel.is_a("IfcRelSpaceBoundary"):
            relating_object = ifc_rel.RelatingSpace
            related_objects = [ifc_rel.RelatedBuildingElement]
        else:
            continue

        for related_object in related_objects:
            if related_objects is not None and relating_object is not None and related_object is not None:
                if relating_object.id() in ifc_objects and related_object.id() in ifc_objects:
                    source_node = Node(
                        id=ifc_objects[relating_object.id()]["id"],
                        type=ifc_objects[relating_object.id()]["type"],
                    )
                    target_node = Node(
                        id=ifc_objects[related_object.id()]["id"],
                        type=ifc_objects[related_object.id()]["type"],
                    )
                    relationships.append(
                        Relationship(
                            source=source_node,
                            target=target_node,
                            type=ifc_rel.is_a(),
                            properties={},
                        )
                    )
                    relationships.append(
                        Relationship(
                            source=source_node,
                            target=target_node,
                            type='IFC',
                            properties={},
                        )
                    )

    #print(nodes) 
    #print(relationships) 

    return [GraphDocument(nodes=nodes, relationships=relationships, source=Document(ifc_file_path))]

# May return IFC2X3, IFC4, or IFC4X3.
#
#wall = model.by_type("IfcWall")[0]
#for wall_type in model.by_type("IfcWallType"):
#    print("The wall type element is", wall_type)
#    print("The name of the wall type is", wall_type.Name)
#
#    # Get all properties and quantities as a dictionary
#    # returns {"Pset_WallCommon": {"id": 123, "FireRating": "2HR", ...}}
#    psets = ifcopenshell.util.element.get_psets(wall_type)
#
#    # Get only properties and not quantities
#    print(psets)
#
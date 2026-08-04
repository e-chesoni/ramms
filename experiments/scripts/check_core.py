"""Manual checks derived from the notebook's chain-creation sections."""

from _fixtures import make_four_unit_chain, make_two_unit_chain


def describe_chain(chain, name: str) -> None:
    print(f"\n{name}:")
    for index, unit in enumerate(chain.units):
        print(f"Unit {index}: {unit.unit_type.value}")

        print("  Nodes:")
        for node in unit.nodes:
            label = getattr(node, "long_name", node.descriptor)
            print(f"    {label}")

        print("  Struts:")
        for strut in unit.struts:
            label = getattr(strut, "long_name", strut)
            print(f"    {label}")


def main() -> None:
    two_unit_chain = make_two_unit_chain()
    four_unit_chain = make_four_unit_chain()

    unit_0 = two_unit_chain.units[0]
    print("Unit type:", unit_0.unit_type)
    print("Node diameter:", unit_0.node_diameter)
    print("Calculated rail length:", unit_0.rail_length)

    print("\nRail nodes:")
    for node in unit_0.rail_nodes:
        print(node.descriptor, node.coordinates)

    print("\nRails:")
    for rail in unit_0.rails:
        print(rail.descriptor, rail.node_1.coordinates, rail.node_2.coordinates)

    describe_chain(four_unit_chain, "Four Unit Chain")
    describe_chain(two_unit_chain, "Two Unit Chain")


if __name__ == "__main__":
    main()

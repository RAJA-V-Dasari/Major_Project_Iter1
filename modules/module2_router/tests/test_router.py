import json

from router.router import RegionRouter


def test_router():

    router = RegionRouter()

    router.load("input/segmentation_output.json")

    router.validate()

    router.sort_regions()

    router.route()

    router.save("outputs/test_output.json")

    with open("outputs/test_output.json") as f:

        data = json.load(f)

    assert len(data["pages"]) == 1

    assert data["pages"][0]["regions"][0]["processor"] == "ocr"
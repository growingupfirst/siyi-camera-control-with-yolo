from typing import Optional, Union

import cv2
import numpy as np

from supervision.annotators.base import BaseAnnotator, ImageType
from supervision.annotators.utils import (
    ColorLookup,
    resolve_color
)
from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import draw_polygon
from supervision.geometry.core import Position
from supervision.utils.conversion import (
    ensure_cv2_image_for_annotation
)


class XAnnotator(BaseAnnotator):
    """
    A class for drawing dots on an image at specific coordinates based on provided
    detections.
    """

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        radius: int = 4,
        position: Position = Position.CENTER,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        outline_thickness: int = 0,
        outline_color: Union[Color, ColorPalette] = Color.BLACK,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating detections.
            radius (int): Radius of the drawn dots.
            position (Position): The anchor position for placing the dot.
            color_lookup (ColorLookup): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
            outline_thickness (int): Thickness of the outline of the dot.
            outline_color (Union[Color, ColorPalette]): The color or color palette to
                use for outline. It is activated by setting outline_thickness to a value
                greater than 0.
        """
        self.color: Union[Color, ColorPalette] = color
        self.radius: int = radius
        self.position: Position = position
        self.color_lookup: ColorLookup = color_lookup
        self.outline_thickness = outline_thickness
        self.outline_color: Union[Color, ColorPalette] = outline_color

    @ensure_cv2_image_for_annotation
    def annotate(
        self,
        scene: ImageType,
        detections: Detections,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> ImageType:
        """
        Annotates the given scene with dots based on the provided detections.

        Args:
            scene (ImageType): The image where dots will be drawn.
                `ImageType` is a flexible type, accepting either `numpy.ndarray`
                or `PIL.Image.Image`.
            detections (Detections): Object detections to annotate.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image, matching the type of `scene` (`numpy.ndarray`
                or `PIL.Image.Image`)

        Example:
            ```python
            import supervision as sv

            image = ...
            detections = sv.Detections(...)

            dot_annotator = sv.DotAnnotator()
            annotated_frame = dot_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            ```

        ![dot-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/dot-annotator-example-purple.png)
        """
        assert isinstance(scene, np.ndarray)
        xy = detections.get_anchors_coordinates(anchor=self.position)
        for detection_idx in range(len(detections)):
            color = resolve_color(
                color=self.color,
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )
            center = (int(xy[detection_idx, 0]), int(xy[detection_idx, 1]))

            cv2.putText(scene, 'x', center, cv2.FONT_HERSHEY_SIMPLEX,0.5, color.as_bgr())
            if self.outline_thickness:
                outline_color = resolve_color(
                    color=self.outline_color,
                    detections=detections,
                    detection_idx=detection_idx,
                    color_lookup=self.color_lookup
                    if custom_color_lookup is None
                    else custom_color_lookup,
                )
                cv2.putText(
                    scene,
                    'x',
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.1,
                    color=outline_color.as_bgr(),
                    thickness=self.outline_thickness,
                )
        return scene

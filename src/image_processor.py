# src/image_processor.py
import os
import base64
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

def convert_images_to_base64(raw_pdf_elements):
    """
    Converts image elements from unstructured raw_pdf_elements to base64.
    Args:
        raw_pdf_elements (list): List of unstructured elements.
    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, each containing image path and base64 encoded data.
            - dict: A dictionary where keys are (page_number, coordinates_tuple) and values are base64 data.
    """
    logger.info(f"Starting conversion of {len(raw_pdf_elements)} elements to base64 images.")
    raw_image_list = []
    image_data_dict = {}

    for element in tqdm(raw_pdf_elements, desc="Converting images to base64"):
        if element.category == 'Image':
            image_path = element.metadata.image_path if hasattr(element.metadata, 'image_path') else None

            if image_path and os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as image_file:
                        image_base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
                        raw_image_list.append({
                            "image_path": image_path,
                            "base64_data": image_base64_encoded,
                            "page_number": element.metadata.page_number,
                            "coordinates": element.metadata.coordinates
                        })
                        image_key = (
                            element.metadata.page_number,
                            tuple(map(tuple, element.metadata.coordinates.points))
                        )
                        image_data_dict[image_key] = image_base64_encoded
                        logger.debug(f"Encoded image from page {element.metadata.page_number} at {image_path}")
                except Exception as e:
                    logger.error(f"Failed to encode image at {image_path} on page {element.metadata.page_number}: {e}")
            else:
                logger.warning(
                    f"Image path not found or invalid for element on page {element.metadata.page_number}"
                )

    logger.info(f"Completed conversion. Total images encoded: {len(raw_image_list)}")
    return raw_image_list, image_data_dict


def find_closest_image(caption, images):
    """
    Finds the closest image to a caption based on page number and coordinates.
    Args:
        caption (unstructured.documents.elements.Element): The caption element.
        images (list): List of image dictionaries (raw_image_list from convert_images_to_base64).
    Returns:
        dict: The closest image dictionary, or None if no suitable image is found.
    """
    logger.debug("Starting find_closest_image function.")
    if not hasattr(caption.metadata, 'coordinates') or not hasattr(caption.metadata, 'page_number'):
        logger.error(f"Caption missing coordinates or page_number: {caption}")
        return None

    caption_page_number = caption.metadata.page_number
    # Ensure coordinates are in a consistent format (e.g., list of tuples or tuple of tuples)
    caption_coords = caption.metadata.coordinates.points if hasattr(caption.metadata.coordinates, 'points') else caption.metadata.coordinates
    logger.debug(f"Caption Details -- Page: {caption_page_number}, Coords: {caption_coords}")

    min_distance = float('inf')
    closest_image = None

    try:
        # Assuming coordinates are [((x1,y1), (x2,y2), (x3,y3), (x4,y4))] or similar bounding box
        # We need to compute the center of the caption's bounding box
        if isinstance(caption_coords, list) and len(caption_coords) > 0:
            x_coords = [p[0] for p in caption_coords]
            y_coords = [p[1] for p in caption_coords]
            caption_center_x = (min(x_coords) + max(x_coords)) / 2
            caption_center_y = (min(y_coords) + max(y_coords)) / 2
        else:
            logger.error(f"Unexpected caption coordinates format: {caption_coords}")
            return None
        logger.debug(f"Caption center: ({caption_center_x}, {caption_center_y})")
    except Exception as e:
        logger.error(f"Error calculating caption center for {caption.metadata}: {e}")
        return None

    found_on_page = 0

    for idx, img_data in enumerate(tqdm(images, desc="Searching for closest image", unit="img", leave=False)):
        if img_data.get('page_number') == caption_page_number and img_data.get('coordinates'):
            found_on_page += 1
            img_coords = img_data['coordinates'] # This should be similar to caption_coords format
            try:
                if isinstance(img_coords, list) and len(img_coords) > 0:
                    x_coords = [p[0] for p in img_coords]
                    y_coords = [p[1] for p in img_coords]
                    img_center_x = (min(x_coords) + max(x_coords)) / 2
                    img_center_y = (min(y_coords) + max(y_coords)) / 2
                else:
                    logger.debug(f"Skipping image[{idx}] due to unexpected coordinates format: {img_coords}")
                    continue

                logger.debug(
                    f"Image[{idx}]: center=({img_center_x}, {img_center_y}), path={img_data.get('image_path')}"
                )

                horizontal_distance = abs(caption_center_x - img_center_x)
                vertical_distance = abs(caption_center_y - img_center_y)
                distance = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5

                logger.debug(
                    f"Image[{idx}]: distance={distance:.2f} (horiz: {horizontal_distance:.2f}, vert: {vertical_distance:.2f})"
                )

                if distance < min_distance:
                    logger.debug(
                        f"Image[{idx}]: New closest (dist {distance:.2f}) [prev min: {min_distance:.2f}]"
                    )
                    min_distance = distance
                    closest_image = img_data

            except Exception as e:
                logger.error(f"Error processing image[{idx}]: {e}")

    if found_on_page == 0:
        logger.warning(f"No images found on the same page ({caption_page_number}) as the caption.")

    if closest_image:
        logger.info(
            f"Closest image to caption on page {caption_page_number}: {closest_image.get('image_path')} "
            f"with distance {min_distance:.2f}"
        )
    else:
        logger.warning(
            f"No suitable image found for caption on page {caption_page_number}!"
        )

    return closest_image


def generate_caption_image_page_number(raw_pdf_elements, image_base64_dict):
    """
    Separates images and captions, creates a mapping of images to captions and base64.
    Args:
        raw_pdf_elements (list): List of unstructured elements.
        image_base64_dict (dict): Dictionary of image_key -> base64_data.
    Returns:
        dict: Mapping image_key to base64_data, caption text, and page number.
    """
    logger.info(f"Extracting images and captions from {len(raw_pdf_elements)} PDF elements.")

    # Filter elements to get actual image elements (not just any element with category 'Image')
    # Use 'FigureCaption' for captions based on Unstructured's common output
    images = [el for el in raw_pdf_elements if el.category == 'Image']
    captions = [el for el in raw_pdf_elements if el.category == 'FigureCaption']


    logger.info(f"Found {len(images)} raw images and {len(captions)} captions in PDF.")

    caption_image_dict = {}

    with logging_redirect_tqdm():
        # tqdm on captions
        for idx, caption in enumerate(tqdm(captions, desc="Mapping captions to images", unit="caption")):
            logger.debug(f"Processing caption[{idx}] page={getattr(caption.metadata, 'page_number', None)}...")

            # For `find_closest_image` to work with the `images` list (which are unstructured elements),
            # we need to extract the relevant metadata into a list of dictionaries.
            images_for_closest_search = [
                {
                    "image_path": el.metadata.image_path if hasattr(el.metadata, 'image_path') else None,
                    "page_number": el.metadata.page_number,
                    "coordinates": el.metadata.coordinates.points if hasattr(el.metadata.coordinates, 'points') else el.metadata.coordinates
                }
                for el in images if hasattr(el.metadata, 'page_number') and hasattr(el.metadata, 'coordinates')
            ]


            closest_image = find_closest_image(caption, images_for_closest_search)
            if closest_image:
                # Ensure coordinates are a tuple of tuples for the key
                # This depends on the exact structure of element.metadata.coordinates.points
                # if it's already a list of (x,y) tuples, then tuple(map(tuple, ...)) is correct.
                # If it's already a tuple of tuples, just use it.
                coords_for_key = closest_image['coordinates']
                if not isinstance(coords_for_key, tuple): # Ensure it's hashable
                    try:
                        coords_for_key = tuple(map(tuple, coords_for_key))
                    except TypeError:
                        logger.error(f"Could not convert coordinates to hashable tuple: {closest_image['coordinates']}")
                        continue

                image_key = (
                    closest_image['page_number'],
                    coords_for_key
                )

                if image_key in image_base64_dict:
                    caption_image_dict[image_key] = {
                        "base64_data": image_base64_dict[image_key],
                        "caption_text": str(caption),
                        "page_number": closest_image['page_number']
                    }
                    logger.debug(f"Matched caption[{idx}] to image_key={image_key}")
                else:
                    logger.warning(
                        f"Matched image key {image_key} not found in base64 dict for page {image_key[0]}"
                    )
            else:
                logger.warning(
                    f"No closest image found for caption on page {getattr(caption.metadata, 'page_number', None)}: {str(caption)[:50]}..."
                )

        # tqdm on images (for completeness: adding images without captions)
        for img_idx, img_element in enumerate(tqdm(images, desc="Adding uncaptioned images", unit="image")):
            if not hasattr(img_element.metadata, 'page_number') or not hasattr(img_element.metadata, 'coordinates'):
                logger.warning(f"Skipping image element {img_idx} due to missing metadata.")
                continue

            # Ensure coordinates are a tuple of tuples for the key
            coords_for_key = img_element.metadata.coordinates.points if hasattr(img_element.metadata.coordinates, 'points') else img_element.metadata.coordinates
            if not isinstance(coords_for_key, tuple): # Ensure it's hashable
                try:
                    coords_for_key = tuple(map(tuple, coords_for_key))
                except TypeError:
                    logger.error(f"Could not convert coordinates for orphaned image to hashable tuple: {coords_for_key}")
                    continue

            img_key_for_dict = (
                img_element.metadata.page_number,
                coords_for_key
            )

            if img_key_for_dict not in caption_image_dict:
                caption_image_dict[img_key_for_dict] = {
                    "base64_data": image_base64_dict.get(img_key_for_dict, ""), # Get from the pre-converted dict
                    "caption_text": "", # No caption for orphaned images
                    "page_number": img_element.metadata.page_number
                }
                logger.info(
                    f"Added orphaned image (no caption) at page {img_element.metadata.page_number} as image_key {img_key_for_dict}"
                )

    logger.info(f"Total image-caption objects generated: {len(caption_image_dict)}")

    return caption_image_dict

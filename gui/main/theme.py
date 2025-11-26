import dearpygui.dearpygui as dpg

BRAND_COLORS = {
    "primary_bg": (248, 245, 235, 255),  # #F8F5EB
    "primary_accent": (213, 219, 67, 255),  # #D5DB43
    "primary_dark": (38, 49, 51, 255),  # #263133
    "secondary_bg": (240, 235, 222, 255),  # #F0EBDE
    "secondary_accent": (188, 191, 91, 255),  # #BCBF5B
    "secondary_dark": (64, 92, 82, 255),  # #405C52
}


def apply_brand_theme() -> None:
    """Apply the project's brand colors to the current DearPyGui context."""

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, BRAND_COLORS["primary_bg"])
            dpg.add_theme_color(dpg.mvThemeCol_Text, BRAND_COLORS["primary_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, BRAND_COLORS["primary_accent"])
            dpg.add_theme_color(dpg.mvThemeCol_Button, BRAND_COLORS["primary_accent"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, BRAND_COLORS["secondary_accent"])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, BRAND_COLORS["secondary_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, BRAND_COLORS["secondary_bg"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, BRAND_COLORS["secondary_accent"])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, BRAND_COLORS["secondary_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_Header, BRAND_COLORS["secondary_bg"])
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, BRAND_COLORS["primary_accent"])
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, BRAND_COLORS["secondary_dark"])
            dpg.add_theme_color(dpg.mvThemeCol_Tab, BRAND_COLORS["secondary_bg"])
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, BRAND_COLORS["primary_accent"])
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, BRAND_COLORS["secondary_dark"])
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)

    dpg.bind_theme(theme)

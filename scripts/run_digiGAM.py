import subprocess
from pathlib import Path

# Define paths
script_dir = Path("../scripts")
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Define groups by platform
groups = {
    "podcast": [
        "1_podcast_ingestion.py",
        "2_podcast_processing.py"
    ],
    "site": [
        "1_site_ingestion.py",
        "2_site_processing.py",
        "3_site_reach.py"
    ],
    "facebook": [
        "1_socialMedia_ingestion_facebook_country.py",
        "1_socialMedia_ingestion_facebook_engagements.py",
        "3_socialMedia_combination_facebook.py",
        "4_socialMedia_processing_facebook.py",
        "5_test_comparingMK_BD_social_fb.py"
    ],
    "instagram": [
        "1_socialMedia_ingestion_instagram_country.py",
        "1_socialMedia_ingestion_instagram_engagements.py",
        "3_socialMedia_combination_instagram.py",
        "4_socialMedia_processing_instagram.py",
        "5_test_comparingMK_BD_social_ig.py"
    ],
    "twitter": [
        "1_socialMedia_ingestion_twitter_country.py",
        "1_socialMedia_ingestion_twitter_engagments.py",
        "3_socialMedia_combination_twitter.py",
        "4_socialMedia_processing_twitter.py",
        "5_test_comparingMK_BD_social_twitter.py"
    ],
    "youtube": [
        "1_socialMedia_ingestion_youtube_analytics.py",
        "1_socialMedia_ingestion_youtube_redshift_new.py",
        "3_socialMedia_combination_youtube.py",
        "4_socialMedia_processing_youtube.py",
        "5_test_comparingMK_BD_social_yt.py"
    ],
    "tiktok": [
        "1_socialMedia_ingestion_tiktok.py",
        "3_socialMedia_combination_tiktok.py",
        "4_socialMedia_processing_tiktok.py",
        "5_test_comparingMK_BD_social_ttk.py"
    ],
    "telegram": [
        "1_socialMedia_ingestion_telegram.py"
    ],
    "rest": [
        "1_socialMedia_ingestion_Rest.py",
        "5_test_comparingMK_BD_social_rest.py"
    ]
}

final_scripts = [
    "6_combining_social.py",
    "5_test_comparingMK_BD_social_WSC.py",
    "8_total_digital.py",
    "5_test_comparingMK_BD_social_css.py"
]

# Track if any group failed
group_failed = False

# Run each group
for group_name, scripts in groups.items():
    print(f"\nRunning group: {group_name}")
    try:
        for script_name in scripts:
            script_path = script_dir / script_name
            log_path = log_dir / (script_path.stem + ".log")
            print(f"Running {script_path}, logging to {log_path}")
            with open(log_path, "w") as log_file:
                subprocess.run(["python3", str(script_path)], stdout=log_file, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Group '{group_name}' failed. Final scripts will be skipped.")
        group_failed = True

# Run final scripts only if all groups succeeded
if not group_failed:
    print("\n✅ All groups succeeded. Running final scripts...")
    for script_name in final_scripts:
        script_path = script_dir / script_name
        log_path = log_dir / (script_path.stem + ".log")
        print(f"Running {script_path}, logging to {log_path}")
        with open(log_path, "w") as log_file:
            subprocess.run(["python3", str(script_path)], stdout=log_file, stderr=subprocess.STDOUT, check=True)
else:
    print("\n⚠️ Skipping final scripts due to earlier failures.")
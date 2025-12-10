import subprocess
from pathlib import Path

# Define paths
#script_dir = Path(__file__).resolve().parent  # Always points to the scripts folder
script_dir = Path.cwd()  # Current working directory
log_dir = Path("../logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Define groups by platform
groups = {
    "facebook": [
        #"1_socialMedia_ingestion_facebook_country_fromNotebook.py",
        #"1_socialMedia_ingestion_facebook_engagements_fromNotebook.py",
        #"3_socialMedia_combination_facebook_fromNotebook.py",
        #"4_socialMedia_processing_facebook_fromNotebook.py",
    ],
    "instagram": [
        #"1_socialMedia_ingestion_instagram_country_fromNotebook.py",
        #"1_socialMedia_ingestion_instagram_engagements_fromNotebook.py",
        #"3_socialMedia_combination_instagram_fromNotebook.py",
        #"4_socialMedia_processing_instagram_fromNotebook.py",
    ],
    "twitter": [
        #"1_socialMedia_ingestion_twitter_country_fromNotebook.py",
        #"1_socialMedia_ingestion_twitter_engagments_fromNotebook.py",
        #"3_socialMedia_combination_twitter_fromNotebook.py",
        #"4_socialMedia_processing_twitter_fromNotebook.py",
    ],
    "youtube": [
        #"1_socialMedia_ingestion_youtube_analytics_fromNotebook.py",
        #"1_socialMedia_ingestion_youtube_redshift_fromNotebook.py",
        #"3_socialMedia_combination_youtube_fromNotebook.py",
        #"4_socialMedia_processing_youtube_fromNotebook.py",
    ],
    "tiktok": [
        #"1_socialMedia_ingestion_tiktok_fromNotebook.py",
        "3_socialMedia_combination_tiktok_fromNotebook.py",
        "4_socialMedia_processing_tiktok_fromNotebook.py",
    ],
    "site": [
        #"1_site_ingestion_fromNotebook.py",
        #"2_site_processing_fromNotebook.py",
        #"3_site_reach_fromNotebook.py"
    ],
    "podcast": [
        #"1_podcast_ingestion_fromNotebook.py",
        #"2_podcast_processing_fromNotebook.py"
    ],
    
    #"telegram": [
    #    "1_socialMedia_ingestion_telegram_fromNotebook.py"
    #],
    #"rest": [
    #    "1_socialMedia_ingestion_Rest_fromNotebook.py",
    #    "5_test_comparingMK_BD_social_rest_fromNotebook.py"
    #]
}

final_scripts = [
    "6_combining_social_fromNotebook.py",
    "8_total_digital_fromNotebook.py",
]

# Track if any group failed
group_failed = False

# Run each group
for group_name in groups.keys():
#for group_name in ['tiktok', 'site']:
    print(f"\nRunning group: {group_name}")
    scripts = groups[group_name]
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
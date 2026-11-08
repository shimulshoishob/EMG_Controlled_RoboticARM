// Copy this file to secrets.h (it is gitignored) and set a long random key.
// The same key is entered in the desktop app (Armband > step 3, or "Import from secrets.h").
// Generate one with:  python3 -c "import secrets; print(secrets.token_hex(24))"
#pragma once
#define DEVICE_ACCESS_KEY_VALUE "CHANGE_THIS_TO_A_LONG_RANDOM_KEY"

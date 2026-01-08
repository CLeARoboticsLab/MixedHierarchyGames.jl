#!/usr/bin/env bash
# Script to fix FreeType compatibility issue with Qt6
# This updates FreeType to 2.13+ which is required by Qt6 artifacts

set -e

echo "Checking FreeType version..."
CURRENT_VERSION=$(dpkg -l | grep libfreetype6 | awk '{print $3}' | cut -d'-' -f1)
echo "Current FreeType version: $CURRENT_VERSION"

if [ -z "$CURRENT_VERSION" ]; then
    echo "FreeType not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y libfreetype6 libfreetype6-dev
    exit 0
fi

# Check if version is 2.13 or higher
MAJOR=$(echo $CURRENT_VERSION | cut -d'.' -f1)
MINOR=$(echo $CURRENT_VERSION | cut -d'.' -f2)

if [ "$MAJOR" -gt 2 ] || ([ "$MAJOR" -eq 2 ] && [ "$MINOR" -ge 13 ]); then
    echo "FreeType version is already 2.13+ ($CURRENT_VERSION). No update needed."
    exit 0
fi

echo "FreeType version $CURRENT_VERSION is too old. Qt6 requires 2.13+."
echo ""
echo "Options:"
echo "1. Build FreeType 2.13+ from source (recommended)"
echo "2. Try to use system libraries with LD_PRELOAD (workaround)"
echo "3. Clear Qt6 artifacts and hope for compatible version (unlikely to work)"
echo ""
echo "For option 1, you can:"
echo "  wget https://download.savannah.gnu.org/releases/freetype/freetype-2.13.2.tar.gz"
echo "  tar -xzf freetype-2.13.2.tar.gz"
echo "  cd freetype-2.13.2"
echo "  ./configure --prefix=/usr/local"
echo "  make"
echo "  sudo make install"
echo "  sudo ldconfig"
echo ""
echo "For option 2 (workaround), try running with:"
echo "  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfreetype.so.6 python3 pursuit_evasion_controller.py"
echo ""
echo "For option 3, clear Qt6 artifacts:"
echo "  rm -rf ~/.julia/artifacts/e58862cd7eac46c8e666f512c310cd1c09a071ba"
echo "  rm -rf ~/.julia/artifacts/61bd4b8dad0917803b344a4de5848bfb298dbf52"


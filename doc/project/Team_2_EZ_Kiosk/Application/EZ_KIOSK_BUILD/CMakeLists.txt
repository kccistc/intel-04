cmake_minimum_required(VERSION 3.21.1)

option(LINK_INSIGHT "Link Qt Insight Tracker library" ON)
option(BUILD_QDS_COMPONENTS "Build design studio components" ON)

project(EZ_KIOSK_CLEANApp LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)

find_package(Qt6 6.2 REQUIRED COMPONENTS Core Gui Qml Quick)

set(SOURCES
    src/main.cpp
    src/connectEvent.cpp
    src/voiceEvent.cpp
    src/burgers.cpp
    src/chicken.cpp
    src/beverage.cpp
)

set(HEADERS
    src/connectEvent.hpp
    src/voiceEvent.hpp
    src/burgers.hpp
    src/chicken.hpp
    src/beverage.hpp
    src/json.hpp
)

if (Qt6_VERSION VERSION_GREATER_EQUAL 6.3)
    qt_standard_project_setup()
endif()

qt_add_executable(EZ_KIOSK_CLEANApp ${SOURCES} ${HEADERS}
    src/chicken.hpp
    src/chicken.cpp
    src/beverage.hpp
    src/beverage.cpp
    src/server.h
    src/server.cpp
    src/worker.h
    src/worker.cpp
    src/voiceQnA.h
    src/voiceQnA.cpp


)

qt_add_resources(EZ_KIOSK_CLEANApp "configuration"
    PREFIX "/"
    FILES
        qtquickcontrols2.conf
)

target_link_libraries(EZ_KIOSK_CLEANApp PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Qml
    Qt6::Quick
)

set(QT_QML_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/qml)

set(QML_IMPORT_PATH ${QT_QML_OUTPUT_DIRECTORY}
    CACHE STRING "Import paths for Qt Creator's code model"
    FORCE
)

if (BUILD_QDS_COMPONENTS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/qmlcomponents)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/qmlmodules)

if (LINK_INSIGHT)
    include(${CMAKE_CURRENT_SOURCE_DIR}/insight)
endif ()

include(GNUInstallDirs)
install(TARGETS EZ_KIOSK_CLEANApp
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# make IDEs aware of the QML import path
set(QML_IMPORT_PATH ${PROJECT_BINARY_DIR}/qml CACHE PATH
    "Path to the custom QML components defined by the project")

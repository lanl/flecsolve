function(flecsolve_headers)
  target_sources(flecsolve PUBLIC FILE_SET public_headers TYPE HEADERS FILES ${ARGN})
endfunction()

function(flecsolve_sources)
  target_sources(flecsolve PRIVATE ${ARGN})
endfunction()

target_sources(flecsolve PUBLIC FILE_SET public_headers TYPE HEADERS)

add_subdirectory(vectors)
add_subdirectory(operators)
add_subdirectory(solvers)
add_subdirectory(physics)
add_subdirectory(time-integrators)
add_subdirectory(util)
add_subdirectory(matrices)

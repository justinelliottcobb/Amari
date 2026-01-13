{ name = "amari-purescript-examples"
, dependencies =
  [ "aff"
  , "arrays"
  , "console"
  , "effect"
  , "either"
  , "exceptions"
  , "foldable-traversable"
  , "foreign"
  , "integers"
  , "maybe"
  , "numbers"
  , "prelude"
  , "refs"
  , "transformers"
  , "tuples"
  , "web-promise"
  ]
, packages = ./packages.dhall
, sources = [ "src/**/*.purs", "test/**/*.purs" ]
}

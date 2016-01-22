class CreateReoccurences < ActiveRecord::Migration
  def change
    create_table :reoccurences do |t|
      t.string :query
      t.integer :interval
      t.string :unit
      t.datetime :time

      t.timestamps null: false
    end
  end
end

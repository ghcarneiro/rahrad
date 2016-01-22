class AddUserIdToReoccurences < ActiveRecord::Migration
  def change
	add_column :reoccurences, :user_id, :integer
  end
end
